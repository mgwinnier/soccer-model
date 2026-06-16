# ⚽ Badass Soccer Model — 2026 World Cup Predictor

A statistically rigorous engine that predicts international football matches and
simulates the **2026 FIFA World Cup** to championship odds. It blends four
complementary, well-established modeling approaches — not one black box — is
**honestly backtested** on past World Cups with proper scoring rules, and compares
itself to **live Vegas odds** in a dashboard with value/edge detection.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py        # data → features → backtest → ablation → sim
streamlit run app/dashboard.py        # interactive UI: matches vs Vegas, odds, sim
python -m src.predict.predict_match "Brazil" "Argentina"
python -m src.predict.value 2026-06-16   # model vs market for a date, with value flags
```

## Optional API keys (everything works without them)

Credentials live **inside the project** (git-ignored) — no system env vars needed.

| Key | Where to put it | Unlocks |
|---|---|---|
| **API-Football** | `.env` → `API_FOOTBALL_KEY=...` (copy `.env.example`) | confirmed lineups + injuries |
| **Kaggle** | `secrets/kaggle.json` | FIFA player ratings |

Verify with `python -m src.data.secrets_check`. The two work **together**:
API-Football says *who is injured*, Kaggle FIFA ratings say *how good they are*, and
`features/availability.py` turns "key player out" into an expected-goals penalty
(`MatchPredictor.predict(..., home_avail=, away_avail=)`) that sharpens imminent-match
predictions. With no keys, the multiplier is 1.0 — no effect, nothing fabricated.

## The dashboard (`streamlit run app/dashboard.py`)
- **Matches** — pick a date → rich cards per upcoming fixture: model vs de-vigged
  **Vegas** across **Match Result, Total Goals (O/U), and Spread**, each with
  **EV%** and a **Kelly stake $**; model **BTTS**; a context strip (Elo, last-5 form,
  xG, head-to-head, injuries); and a **scoreline heatmap**. Live odds from ESPN.
- **Value Board** — every +EV bet across all matches ranked by EV, with Kelly stakes
  **capped to your bankroll**, CSV export, and an honest calibration caveat.
- **Tournament** — 2026 championship & advancement odds (Monte-Carlo).
- **Performance** — backtest RPS table, calibration curve, ablation results.
- **Team explorer** — strength, O/U ladder, heatmap + any head-to-head predictor.

**Staking controls** (sidebar): bankroll `$`, **Kelly fraction** slider (default
half-Kelly), min-EV threshold, and a max-exposure cap. EV uses the *offered* price;
the "Vegas" column is the de-vigged *fair* probability so the edge is transparent.
Betting math lives in `src/predict/betting.py` (`expected_value`, `kelly_fraction`,
`kelly_stake`); per-market evaluation + the Value Board in `src/predict/value.py`.

### Calibration, market-anchoring & CLV (the credible-edge layer)
- **Per-market calibration** (`models/market_calibration.py`) — Totals/Spread/BTTS
  probabilities are de-biased with isotonic maps fit leak-free on 32k matches
  (`backtest/markets_backtest.py` exposes the bias). Applied in `analyze()`.
- **Market anchoring** (`predict/anchor.py`) — the calibrated model is blended toward
  the de-vigged sharp price (`w=0.5` default), shrinking edges to credible size.
- **Closing-line value** (`predict/clv.py`) — `snapshot` logs each recommended bet at
  the offered price; `grade` settles it vs the **closing** line + result → `clv_ledger.csv`.
  Beating the close is the real edge test; the dashboard's **CLV** page tracks it.
  `scripts/snapshot_odds.py` runs the daily grade+snapshot (schedule it with Task Scheduler).
- **Grade past picks**: `python -m src.predict.bet_grade 2026-06-11` replays the model's
  recommendations on already-played matches at real closing prices (3 modes: independent /
  calibrated / calibrated+anchored).
- **Line movement**: ESPN opening→closing prices give a sharp-money signal (shown on cards).
- **Tuned GBM** (`models/gbm_tune.py`, Optuna) + **current EA FC24 player ratings**
  (was a stale 2018 snapshot). A **bivariate-Poisson** model exists but the ablation found
  λ3≈0 (goals uncorrelated), so Dixon-Coles stays the default — kept honest, not bolted on.

> ⚠ **Honest caveat:** the model's 1X2 calibration is backtested (~0.20 RPS), but its
> **totals/spread/longshot** prices are *not* — those are derived from the same
> scoreline distribution and shown without a market-validated edge. EV/Kelly are only
> as good as the model's probabilities; treat large EVs on longshots with skepticism.
> Not betting advice.

## Results at a glance

**Backtest** (walk-forward, pooled over the 2010–2022 World Cups, 256 matches —
lower RPS is better):

| Model | RPS | Log-loss | Accuracy |
|---|---:|---:|---:|
| Dixon-Coles | **0.2022** | 0.980 | 54.7% |
| **Ensemble** | 0.2026 | 0.983 | 54.7% |
| Elo | 0.2063 | 0.992 | 55.1% |
| LightGBM | 0.2076 | 1.008 | 52.3% |
| Home-prior baseline | 0.2390 | 1.073 | 41.4% |
| Climatology floor | 0.2437 | 1.087 | 41.4% |

Every model beats the information-free baselines by a wide margin, and the pooled
RPS (~0.20) is in the bookmaker-competitive range for international football. The
ensemble's predicted probabilities are **well calibrated** (see
`reports/calibration.csv`).

**2026 forecast** (top of the table — full table in `reports/wc2026_forecast.csv`):

| Team | Champion |
|---|---:|
| Argentina | 14.5% |
| Spain | 10.9% |
| England | 9.1% |
| Germany | 6.0% |
| Morocco | 5.4% |
| Brazil | 5.2% |

*(Exact numbers move as group-stage results come in — the simulator locks in
matches already played and re-simulates the rest.)*

## How it works

```
public data ──► clean & normalize ──► features ──► models ──► backtest
                                                      │
                                                      └──► Monte-Carlo 2026 sim
```

### Data (all public, auto-downloaded)
- **Match spine** — [martj42/international_results](https://github.com/martj42/international_results):
  ~49k men's internationals, 1872→2026, raw GitHub CSV (no auth). **Required.**
- **Live odds + fixtures** — **ESPN hidden API** (`src/data/odds.py`): fixtures +
  bookmaker moneyline in one call, free, no key. De-vigged to true probabilities.
- **xG** — **StatsBomb open data** (`src/data/statsbomb.py`): per-shot xG for WC
  2018/2022 (128 matches). Key-less and reliable. *FBref (`src/data/fbref.py`) is a
  fallback but is usually IP/CAPTCHA-blocked.*
- **Squad values** — Transfermarkt 2026 squads (best-effort scrape; static fallback).
- **Lineups/injuries** — API-Football (`src/data/lineups.py`), **optional, key-gated**
  via `API_FOOTBALL_KEY`; dormant (never fabricated) without a key.

Team names are reconciled to a canonical spelling across all sources
(`src/data/team_names.py`).

### Models (4-member ensemble)
- **Dixon-Coles bivariate Poisson** (`models/dixon_coles.py`) — attack/defense +
  home advantage with the low-score correction and time decay; full scoreline
  distributions. Strongest single model at World Cups.
- **xG-informed Dixon-Coles** (`models/xg_dixon_coles.py`) — same machinery, target
  blended with xG where available (de-noises ratings on elite-tournament matches).
- **Elo** (`features/elo.py`, `models/elo_model.py`) — World-Football-style ratings.
- **LightGBM** (`models/gbm.py`) — gradient boosting over the full engineered feature
  set (Elo, form, squad value, travel/altitude, head-to-head, **xG style ratings**, …).
- **Ensemble** (`models/ensemble.py`) — blends all four via **out-of-fold,
  time-series cross-fitted** weights chosen to minimize RPS. The OOF design avoids the
  train/serve mismatch that makes naïve stacking underperform.
- **Penalty-shootout model** (`models/shootout.py`) — fit on 542 historical shootouts;
  resolves simulated knockout draws by strength instead of a coin flip.

### Model vs market (`src/predict/value.py`)
For each upcoming fixture: model W/D/L vs **de-vigged ESPN/DraftKings** probabilities,
the per-outcome **edge**, whether the model **agrees** with the market favourite, and
**value** legs where the model beats the market by a threshold. Framed as model
disagreement, **not** a profit promise — there is deliberately no historical ROI claim.

### Honesty guarantees
- **No leakage** — every feature uses only pre-match information; backtests freeze
  all data at each tournament's start. Post-match xG is used *only* as a training
  target / for trailing as-of ratings, never as a same-match feature.
- **Proper scoring + ablation** — Ranked Probability Score (primary), log-loss, Brier,
  calibration tables, and an **ablation study** (`backtest/ablation.py`) that keeps a
  feature/model block only if it actually lowers RPS. Unknown teams are **rejected**
  (with fuzzy suggestions), never silently treated as average.
- **Stated approximations** — xG exists for only WC/Euro matches (sparse, NaN
  elsewhere); squad value applies to recent matches only; the 2026 knockout bracket
  uses a structurally-faithful seeded bracket rather than FIFA's exact 495-row
  third-place table (the group stage, which drives most of the result, is exact).

## Usage

```bash
pip install -r requirements.txt

# Full pipeline: download → clean → features → backtest → simulate → report
python scripts/run_pipeline.py

# Re-run without re-downloading or re-backtesting
python scripts/run_pipeline.py --skip-download --skip-backtest --iterations 50000

# Predict a single match (neutral venue by default)
python -m src.predict.predict_match "Brazil" "Argentina"
python -m src.predict.predict_match "Mexico" "South Korea" --home-advantage

# Individual stages
python -m src.data.download
python -m src.features.build
python -m src.backtest.walkforward
python -m src.simulate.tournament

# Tests
python -m pytest tests/ -q
```

Outputs land in `reports/`: `wc2026_forecast.csv`, `backtest_pooled.csv`,
`backtest_by_worldcup.csv`, `calibration.csv`, and a combined `REPORT.md`.

## Configuration

All knobs live in `config.yaml` — data filters and match-importance weights, Elo
K-factor/home-advantage, Dixon-Coles time-decay (`xi`) and goal truncation,
ensemble members/calibration, the backtest World Cup list, and simulation
iteration count/seed.

## Project layout

```
src/
  data/       download, clean, team-name & geography normalization
  features/   elo · rolling form · squad value · context  →  build.py
  models/     dixon_coles · elo_model · gbm · ensemble  (+ base.py interface)
  backtest/   walkforward · metrics (RPS/log-loss/Brier) · benchmarks
  simulate/   bracket_2026 (groups) · tournament (Monte-Carlo)
  predict/    predict_match (CLI) · report (Markdown)
scripts/      run_pipeline.py
tests/        elo · dixon-coles · metrics · simulator/tiebreakers
```

## Extending it
- **Exact bracket** — drop FIFA's R32 slot tree into `KNOCKOUT_SLOTS` in
  `simulate/bracket_2026.py`.
- **xG features** — wire StatsBomb/FBref into a new `features/` module; the GBM
  picks up any numeric column automatically.
- **Market benchmark** — add de-vigged closing odds to `backtest/benchmarks.py`
  to measure edge.
