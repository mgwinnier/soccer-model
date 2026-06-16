"""Monte-Carlo simulator for the 2026 World Cup.

Engine choice: the **Dixon-Coles goals model** drives the simulation. It was the
single strongest model at past World Cups in our backtest (RPS 0.2022) and, being
a *scoreline* model, it natively produces the goals needed for group tables. It
is fit on all data up to the cutoff date; matches already played in 2026 are
locked to their real results, and only the remainder is simulated.

Vectorisation: every match across all N iterations is sampled at once with
NumPy. Group standings use a composite sort key (points → goal difference →
goals for → Elo as the final deterministic tiebreak). The knockout stage fills a
balanced 32-seed bracket (winners > runners-up > best-8 thirds, ranked by
performance) and resolves each round with vectorised win-probability draws, where
a team's knockout win probability is ``P(win) + ½·P(draw)`` (extra-time/penalties
split draws evenly).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from ..config import load_config, path_for, ensure_dirs
from ..data.clean import load_matches
from ..models.dixon_coles import DixonColesModel
from ..models.shootout import ShootoutModel
from ..models.base import scoreline_to_outcome_probs
from .bracket_2026 import GROUPS, HOST_TEAMS, all_teams, load_played_results

# Standard single-elimination seeding order for 32 seeds (1 = top seed).
_BRACKET32 = [
    1, 32, 16, 17, 8, 25, 9, 24, 4, 29, 13, 20, 5, 28, 12, 21,
    2, 31, 15, 18, 7, 26, 10, 23, 3, 30, 14, 19, 6, 27, 11, 22,
]


class TournamentSimulator:
    def __init__(self, cfg: dict | None = None, as_of: str | None = None,
                 live: bool = False):
        self.cfg = cfg or load_config()
        # live=True: strength (DC + Elo) reflects the freshest 2026 results so a team
        # that overperforms is rated stronger for its remaining (unplayed) matches.
        # Default (False) keeps the frozen-batch path used by all backtests/tests.
        self._live = live and not as_of
        if self._live:
            from .live_state import live_match_frame
            matches = self._live_frame = live_match_frame(self.cfg)
        else:
            matches = load_matches(self.cfg)
            if as_of:
                matches = matches[matches["date"] <= pd.Timestamp(as_of)]
        self.dc = DixonColesModel.from_config(self.cfg).fit(matches)
        self.teams = all_teams()
        self.gidx = {t: i for i, t in enumerate(self.teams)}
        self._load_elo()
        self.shootout = ShootoutModel.fit_from_history(self._elo_dict, self.cfg)
        self._precompute_tables()

    def _load_elo(self) -> None:
        if getattr(self, "_live", False):
            from .live_state import live_elo
            ratings = live_elo(self.cfg, frame=getattr(self, "_live_frame", None))
        else:
            import json
            path = path_for("data_processed", self.cfg) / "elo_ratings.json"
            ratings = json.load(open(path, encoding="utf-8")) if path.exists() else {}
        self._elo_dict = ratings
        self.elo = np.array([ratings.get(t, 1500.0) for t in self.teams])

    # ------------------------------------------------------------ precompute
    def _precompute_tables(self) -> None:
        n = len(self.teams)
        atk = np.array([self.dc.attack_[self.dc._tidx[t]] for t in self.teams])
        dfn = np.array([self.dc.defense_[self.dc._tidx[t]] for t in self.teams])
        c = self.dc.intercept_
        # lam_neutral[a, b] = expected goals of a vs b at a neutral venue
        self.lam = np.exp(c + atk[:, None] + dfn[None, :])
        self.home_mult = float(np.exp(self.dc.home_adv_))
        # Knockout advance prob: win in 90'/ET, else win the shootout.
        # WP[a, b] = P(a wins) + P(draw) * P(a wins shootout | Elo gap)
        so = self.shootout.prob_matrix(self.elo)
        self.wp = np.full((n, n), 0.5)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                mat = self.dc.scoreline_matrix(self.lam[a, b], self.lam[b, a])
                pH, pD, pA = scoreline_to_outcome_probs(mat)
                self.wp[a, b] = pH + pD * so[a, b]

    # ---------------------------------------------------------------- sample
    def _sample_goals(self, a: int, b: int, host: int | None, n: int, rng):
        """Sample (goals_a, goals_b) arrays of length n for a vs b."""
        la, lb = self.lam[a, b], self.lam[b, a]
        if host == a:
            la = la * self.home_mult
        elif host == b:
            lb = lb * self.home_mult
        return rng.poisson(la, n), rng.poisson(lb, n)

    # ------------------------------------------------------------------- run
    def run(self, n_iter: int | None = None, seed: int | None = None,
            played: dict | None = None) -> pd.DataFrame:
        n = n_iter or self.cfg["simulation"]["n_iterations"]
        seed = seed if seed is not None else self.cfg["simulation"]["seed"]
        rng = np.random.default_rng(seed)
        played = load_played_results(self.cfg) if played is None else played
        nT = len(self.teams)

        # per-team round appearance counters
        cnt = {k: np.zeros(nT) for k in ["adv", "r16", "qf", "sf", "final", "champ"]}
        win_group = np.zeros(nT)

        # ---- group stage (vectorised over iterations) ----
        winners = np.empty((n, 12), dtype=int)
        runners = np.empty((n, 12), dtype=int)
        thirds = np.empty((n, 12), dtype=int)
        win_score = np.empty((n, 12))
        run_score = np.empty((n, 12))
        third_score = np.empty((n, 12))

        for gi, (gname, gteams) in enumerate(GROUPS.items()):
            gidx = [self.gidx[t] for t in gteams]
            pts = np.zeros((n, 4)); gd = np.zeros((n, 4)); gf = np.zeros((n, 4))
            local = {t: k for k, t in enumerate(gidx)}
            host = next((t for t in gidx if self.teams[t] in HOST_TEAMS), None)
            for x, y in combinations(gidx, 2):
                key = frozenset((self.teams[x], self.teams[y]))
                if key in played:
                    hteam, hg, ag = played[key]
                    if self.teams[x] == hteam:
                        gx = np.full(n, hg); gy = np.full(n, ag)
                    else:
                        gx = np.full(n, ag); gy = np.full(n, hg)
                else:
                    gx, gy = self._sample_goals(x, y, host, n, rng)
                kx, ky = local[x], local[y]
                gf[:, kx] += gx; gf[:, ky] += gy
                gd[:, kx] += gx - gy; gd[:, ky] += gy - gx
                pts[:, kx] += np.where(gx > gy, 3, np.where(gx == gy, 1, 0))
                pts[:, ky] += np.where(gy > gx, 3, np.where(gy == gx, 1, 0))

            elo4 = np.array([self.elo[t] for t in gidx])
            comp = pts * 1e6 + gd * 1e3 + gf * 10 + elo4[None, :] * 1e-3
            order = np.argsort(-comp, axis=1)  # best -> worst
            g_arr = np.array(gidx)
            winners[:, gi] = g_arr[order[:, 0]]
            runners[:, gi] = g_arr[order[:, 1]]
            thirds[:, gi] = g_arr[order[:, 2]]
            win_score[:, gi] = np.take_along_axis(comp, order[:, :1], axis=1)[:, 0]
            run_score[:, gi] = np.take_along_axis(comp, order[:, 1:2], axis=1)[:, 0]
            third_score[:, gi] = np.take_along_axis(comp, order[:, 2:3], axis=1)[:, 0]
            win_group[g_arr] += 0  # placeholder; counted below
            np.add.at(win_group, winners[:, gi], 1)

        # ---- best 8 third-placed teams ----
        third_order = np.argsort(-third_score, axis=1)
        top8 = third_order[:, :8]
        thirds_q = np.take_along_axis(thirds, top8, axis=1)
        thirds_q_score = np.take_along_axis(third_score, top8, axis=1)

        # ---- seed the 32-team bracket (winners > runners > thirds, by score) ----
        w_ord = np.argsort(-win_score, axis=1)
        r_ord = np.argsort(-run_score, axis=1)
        t_ord = np.argsort(-thirds_q_score, axis=1)
        seed_team = np.concatenate([
            np.take_along_axis(winners, w_ord, axis=1),
            np.take_along_axis(runners, r_ord, axis=1),
            np.take_along_axis(thirds_q, t_ord, axis=1),
        ], axis=1)  # (n, 32), column j == seed rank j+1

        # arrange into bracket order
        pos = [s - 1 for s in _BRACKET32]
        parts = seed_team[:, pos]  # (n, 32) in bracket order

        # advancement to knockout
        np.add.at(cnt["adv"], seed_team.ravel(), 1)

        # ---- knockout rounds (vectorised) ----
        round_keys = ["r16", "qf", "sf", "final", "champ"]
        cur = parts
        for r, key in zip(range(5), round_keys):
            k = cur.shape[1] // 2
            a = cur[:, 0::2]; b = cur[:, 1::2]
            p_a = self.wp[a, b]
            draw = rng.random((n, k))
            nxt = np.where(draw < p_a, a, b)
            np.add.at(cnt[key], nxt.ravel(), 1)
            cur = nxt

        # ---- assemble summary ----
        t2g = {t: g for g, ts in GROUPS.items() for t in ts}
        rows = []
        for i, team in enumerate(self.teams):
            rows.append({
                "team": team, "group": t2g[team],
                "elo": round(float(self.elo[i]), 0),
                "win_group": win_group[i] / n,
                "advance": cnt["adv"][i] / n,
                "reach_r16": cnt["r16"][i] / n,
                "reach_qf": cnt["qf"][i] / n,
                "reach_sf": cnt["sf"][i] / n,
                "reach_final": cnt["final"][i] / n,
                "champion": cnt["champ"][i] / n,
            })
        df = pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)
        df.attrs["n_iter"] = n
        return df


def run_simulation(cfg: dict | None = None, n_iter: int | None = None,
                   write: bool = True) -> pd.DataFrame:
    cfg = cfg or load_config()
    sim = TournamentSimulator(cfg)
    df = sim.run(n_iter)
    if write:
        ensure_dirs(cfg)
        out = path_for("reports", cfg) / "wc2026_forecast.csv"
        df.to_csv(out, index=False)
        print(f"[simulate] {df.attrs['n_iter']:,} iterations -> {out}")
    return df


if __name__ == "__main__":
    df = run_simulation()
    pd.set_option("display.width", 200)
    show = df.head(16).copy()
    for c in ["win_group", "advance", "reach_qf", "reach_sf", "reach_final", "champion"]:
        show[c] = (show[c] * 100).round(1)
    print(show[["team", "group", "elo", "advance", "reach_qf", "reach_sf",
                "reach_final", "champion"]].to_string(index=False))
