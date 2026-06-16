"""Single-match predictor: W/D/L, expected goals, and most-likely scorelines.

Uses the Dixon-Coles goals model (for the full scoreline distribution) blended
with the Elo model (for a second strength signal), both of which need only team
names + venue — so any hypothetical fixture can be predicted without rebuilding
the feature pipeline. Run:

    python -m src.predict.predict_match "Brazil" "Argentina" --neutral
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..config import load_config, path_for
from ..data.clean import load_matches
from ..data.team_names import normalize_team
from ..models.dixon_coles import DixonColesModel
from ..models.elo_model import EloModel
from ..models.base import OUTCOMES
from ..models import wc_goals


class MatchPredictor:
    def __init__(self, cfg: dict | None = None, live: bool = False):
        self.cfg = cfg or load_config()
        # live=True: DC + Elo strength reflect the freshest 2026 results (dashboard
        # use). Default keeps the frozen-batch path for reproducible predictions.
        if live:
            from ..simulate.live_state import live_match_frame, live_elo
            matches = live_match_frame(self.cfg)
            self.ratings = live_elo(self.cfg, frame=matches)
        else:
            matches = load_matches(self.cfg)
            import json
            rpath = path_for("data_processed", self.cfg) / "elo_ratings.json"
            self.ratings = json.load(open(rpath, encoding="utf-8"))
        self._matches = matches            # kept for form / head-to-head context
        self.dc = DixonColesModel.from_config(self.cfg).fit(matches)
        feats = pd.read_parquet(path_for("data_processed", self.cfg) / "features.parquet")
        self._features = feats
        self.elo_model = EloModel(cfg=self.cfg).fit(feats)
        # Canonical set of teams the model actually knows (have real matches).
        self.known_teams = set(self.dc.teams_)
        # Per-market calibrators (de-bias Totals/Spread/BTTS). Empty -> no-op.
        from ..models.market_calibration import load_default as _load_cal
        self.calibrators = _load_cal(self.cfg)
        # World-Cup goals correction (favorite/underdog scales). See models/wc_goals.py.
        self._wc_fav, self._wc_dog = wc_goals.load_scales(self.cfg)

    def _validate(self, name: str) -> str:
        """Normalize and confirm the team is real; else raise with suggestions."""
        canon = normalize_team(name)
        if canon in self.known_teams:
            return canon
        import difflib
        hint = difflib.get_close_matches(canon or str(name), self.known_teams, n=3)
        suffix = f" Did you mean: {', '.join(hint)}?" if hint else ""
        raise ValueError(f"Unknown team '{name}'.{suffix}")

    def _compute(self, home: str, away: str, neutral: bool,
                 home_avail: float, away_avail: float):
        """Shared core: returns (blend probs, lam, mu, scoreline matrix)."""
        from ..models.base import scoreline_to_outcome_probs
        lam, mu = self.dc.expected_goals(home, away, neutral)
        lam, mu = lam * home_avail, mu * away_avail
        # World-Cup scoring-environment correction. The model is trained on all
        # internationals (qualifiers/friendlies included) and under-projects WC goals vs
        # ACTUAL results — and not uniformly: the FAVORITE is under-projected more than
        # the underdog (backtest across 7 WCs). Scaling the favored/underdog sides by
        # their separate historical actual/model ratios zeroes the bias AND improves
        # pooled WC RPS over a flat scale. Calibrated to real scores, not the Vegas line.
        # See src/models/wc_goals.py + src/backtest/wc_goals_backtest.py.
        lam, mu = wc_goals.correct(lam, mu, self._wc_fav, self._wc_dog)
        mat = self.dc.scoreline_matrix(lam, mu)
        dc_p = np.array(scoreline_to_outcome_probs(mat))
        elo_diff = self.ratings.get(home, 1500) - self.ratings.get(away, 1500)
        elo_fx = pd.DataFrame([{"elo_diff": elo_diff, "neutral": neutral}])
        elo_p = self.elo_model.predict_proba(elo_fx)[0]
        blend = (dc_p + elo_p) / 2
        blend = blend / blend.sum()
        # Favorite-longshot calibration: map each W/D/L prob to its observed frequency
        # (the model under-rates heavy favorites and over-rates longshots), renormalize.
        if self.calibrators is not None and self.calibrators.models.get("mr") is not None:
            cal3 = np.array([self.calibrators.calibrate("mr", float(x)) for x in blend])
            if cal3.sum() > 0:
                blend = cal3 / cal3.sum()
        return blend, lam, mu, mat

    def predict(self, home: str, away: str, neutral: bool = True,
                home_avail: float = 1.0, away_avail: float = 1.0) -> dict:
        """Predict a match. ``home_avail``/``away_avail`` are availability
        multipliers (<=1.0) from injuries; 1.0 = full-strength (default)."""
        home, away = self._validate(home), self._validate(away)
        blend, lam, mu, mat = self._compute(home, away, neutral, home_avail, away_avail)
        scores = [
            ((i, j), float(mat[i, j]))
            for i in range(mat.shape[0]) for j in range(mat.shape[1])
        ]
        scores.sort(key=lambda kv: -kv[1])
        return {
            "home": home, "away": away, "neutral": neutral,
            "probs": dict(zip(OUTCOMES, blend.round(4))),
            "expected_goals": (round(lam, 2), round(mu, 2)),
            "top_scorelines": [(f"{i}-{j}", round(p, 3)) for (i, j), p in scores[:6]],
            "over_2_5": round(float(_over_prob(mat, 2.5)), 3),
            "btts": round(float(mat[1:, 1:].sum()), 3),
        }

    def analyze(self, home: str, away: str, neutral: bool = True,
                market_total: float = 2.5, spread_home_line: float | None = None,
                home_avail: float = 1.0, away_avail: float = 1.0) -> dict:
        """Rich analysis: W/D/L + scoreline matrix + O/U ladder + spread cover +
        BTTS + team context. Built on the same scoreline distribution as predict()
        (expected goals already corrected for the WC scoring environment via
        ``WC_GOALS_SCALE`` in ``_compute``)."""
        home, away = self._validate(home), self._validate(away)
        blend, lam, mu, mat = self._compute(home, away, neutral, home_avail, away_avail)
        scores = sorted(
            (((i, j), float(mat[i, j])) for i in range(mat.shape[0])
             for j in range(mat.shape[1])), key=lambda kv: -kv[1])
        cal = self.calibrators  # de-bias derived markets (no-op if unfit)
        ladder = {ln: round(float(cal.calibrate("over", _over_prob(mat, ln))), 4)
                  for ln in (0.5, 1.5, 2.5, 3.5, 4.5)}
        p_over = float(cal.calibrate("over", _over_prob(mat, market_total)))
        out = {
            "home": home, "away": away, "neutral": neutral,
            "probs": {o: float(p) for o, p in zip(OUTCOMES, blend)},
            "expected_goals": (round(lam, 2), round(mu, 2)),
            "scoreline_matrix": mat,
            "top_scorelines": [(f"{i}-{j}", round(p, 4)) for (i, j), p in scores[:8]],
            "ou_ladder": ladder,
            "p_over_market": round(p_over, 4),
            "p_under_market": round(1 - p_over, 4),
            "market_total": market_total,
            "btts": round(float(cal.calibrate("btts", float(mat[1:, 1:].sum()))), 4),
            "home_context": self.team_context(home),
            "away_context": self.team_context(away),
            "h2h": self.head_to_head(home, away),
        }
        if spread_home_line is not None:
            ph_raw, push, _ = _cover_prob(mat, spread_home_line)
            ph = float(cal.calibrate("cover", ph_raw))
            pa = 1.0 - ph - push
            out["spread"] = {
                "home_line": spread_home_line,
                "p_home_cover": round(ph, 4), "p_push": round(push, 4),
                "p_away_cover": round(pa, 4),
            }
        return out

    # ----------------------------------------------------------- context
    def team_context(self, team: str) -> dict:
        """Elo, last-5 form, and latest xG rating for a team."""
        team = normalize_team(team)
        m = self._matches
        played = m[(m["home_team"] == team) | (m["away_team"] == team)].tail(5)
        form, gf, ga = [], 0, 0
        for r in played.itertuples(index=False):
            if r.home_team == team:
                f, a = r.home_score, r.away_score
            else:
                f, a = r.away_score, r.home_score
            gf += f; ga += a
            form.append("W" if f > a else ("D" if f == a else "L"))
        xgf = np.nan
        feat = self._features
        fteam = feat[(feat["home_team"] == team) | (feat["away_team"] == team)]
        if len(fteam) and "home_xgf" in feat.columns:
            last = fteam.iloc[-1]
            xgf = last["home_xgf"] if last["home_team"] == team else last["away_xgf"]
        return {
            "elo": round(float(self.ratings.get(team, 1500)), 0),
            "form": "".join(form), "gf5": int(gf), "ga5": int(ga),
            "xg_rating": (round(float(xgf), 2) if pd.notna(xgf) else None),
        }

    def head_to_head(self, home: str, away: str, n: int = 5) -> dict:
        """Recent head-to-head record between the two teams."""
        home, away = normalize_team(home), normalize_team(away)
        m = self._matches
        mask = (((m["home_team"] == home) & (m["away_team"] == away))
                | ((m["home_team"] == away) & (m["away_team"] == home)))
        h2h = m[mask].tail(n)
        hw = aw = dr = 0
        recent = []
        for r in h2h.itertuples(index=False):
            hs, as_ = r.home_score, r.away_score
            winner = r.home_team if hs > as_ else (r.away_team if as_ > hs else None)
            if winner == home:
                hw += 1
            elif winner == away:
                aw += 1
            else:
                dr += 1
            recent.append(f"{r.home_team[:3]} {hs}-{as_} {r.away_team[:3]}")
        return {"home_wins": hw, "away_wins": aw, "draws": dr,
                "n": len(h2h), "recent": recent}


def _over_prob(mat: np.ndarray, line: float) -> float:
    k = mat.shape[0]
    total = 0.0
    for i in range(k):
        for j in range(k):
            if i + j > line:
                total += mat[i, j]
    return total


def _cover_prob(mat: np.ndarray, home_line: float) -> tuple[float, float, float]:
    """(P home covers, P push, P away covers) for a home handicap ``home_line``.

    Home covers when (home_margin + home_line) > 0; a push (only on integer lines)
    is when it equals exactly 0."""
    k = mat.shape[0]
    home_cover = push = away_cover = 0.0
    for i in range(k):
        for j in range(k):
            adj = (i - j) + home_line
            if adj > 1e-9:
                home_cover += mat[i, j]
            elif adj < -1e-9:
                away_cover += mat[i, j]
            else:
                push += mat[i, j]
    return home_cover, push, away_cover


def _format(p: dict) -> str:
    venue = "neutral" if p["neutral"] else f"{p['home']} home"
    lines = [
        f"\n{p['home']}  vs  {p['away']}   ({venue})",
        "-" * 46,
        f"  {p['home']} win : {p['probs']['H']*100:5.1f}%",
        f"  Draw       : {p['probs']['D']*100:5.1f}%",
        f"  {p['away']} win : {p['probs']['A']*100:5.1f}%",
        f"  Expected goals: {p['expected_goals'][0]} - {p['expected_goals'][1]}",
        f"  Over 2.5: {p['over_2_5']*100:.1f}%   Both teams score: {p['btts']*100:.1f}%",
        "  Most likely scorelines: "
        + ", ".join(f"{s} ({pr*100:.1f}%)" for s, pr in p["top_scorelines"][:5]),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict a single international match.")
    ap.add_argument("home")
    ap.add_argument("away")
    ap.add_argument("--home-advantage", action="store_true",
                    help="treat as a home game for the first team (default neutral)")
    args = ap.parse_args()
    predictor = MatchPredictor()
    try:
        p = predictor.predict(args.home, args.away, neutral=not args.home_advantage)
    except ValueError as exc:
        raise SystemExit(f"\nError: {exc}\n")
    print(_format(p))


if __name__ == "__main__":
    main()
