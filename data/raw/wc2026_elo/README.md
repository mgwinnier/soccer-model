# 2026 FIFA World Cup — Historical Elo Ratings

Pre-tournament Elo ratings and complete historical career data for all **48 teams** qualified for the 2026 FIFA World Cup (USA / Canada / Mexico, June 11 – July 19, 2026). Each row is one team's end-of-year snapshot. Coverage spans **1901 → 2026** — 125 years of international football.

## What's in the file

`elo_ratings_wc2026.csv` — 4,683 rows, 22 columns, UTF-8, comma-separated.

- One row per `(team, year-end snapshot)`.
- Plus one **live snapshot row per team** at the file's creation date (`snapshot_date` ≠ Dec 31) so you always have an as-of-today reading alongside the year-end series.
- Predecessor federations bridged into their modern successors (see *Predecessor handling* below) so each of the 48 teams has the longest continuous history that FIFA recognizes.

## Column dictionary

| Column | Type | Description |
|---|---|---|
| `year` | int | Calendar year of the snapshot. |
| `snapshot_date` | date | `YYYY-12-31` for year-end rows; the scrape date for the live-snapshot row. |
| `country` | string | Modern team name (e.g. *Czechia*, not *Czechoslovakia*; *DR Congo*, not *Zaire*). |
| `country_code` | string | Two-letter team code as used by eloratings.net (mostly ISO 3166-1 alpha-2 but with site-specific quirks like `EN` for England, `SQ` for Scotland, `KR` for South Korea). |
| `rank` | int | Global Elo rank at the snapshot date. |
| `rating` | int | Elo rating at the snapshot date. |
| `rank_max` | int | Best (lowest-numbered) rank ever achieved by the team up to and including this snapshot. |
| `rating_max` | int | Highest Elo rating ever achieved up to this snapshot. |
| `rank_avg` | int | Career-average rank up to this snapshot. |
| `rating_avg` | int | Career-average Elo rating up to this snapshot. |
| `rank_min` | int | Worst (highest-numbered) rank ever held up to this snapshot. |
| `rating_min` | int | Lowest Elo rating ever held up to this snapshot. |
| `matches_total` | int | Cumulative full internationals played. |
| `matches_home` | int | Of `matches_total`, how many were played at home. |
| `matches_away` | int | Of `matches_total`, how many were played away. |
| `matches_neutral` | int | Of `matches_total`, how many were played at a neutral venue. |
| `wins` | int | Cumulative wins. |
| `losses` | int | Cumulative losses. |
| `draws` | int | Cumulative draws. |
| `goals_for` | int | Cumulative goals scored. |
| `goals_against` | int | Cumulative goals conceded. |
| `confederation` | string | UEFA / CONMEBOL / CONCACAF / CAF / AFC / OFC. |
| `is_host` | int | `1` if the team is hosting the tournament (USA, Canada, Mexico), else `0`. |

**Note on "career-to-date" fields.** `rank_max`, `rating_max`, `rank_avg`, `rating_avg`, `rank_min`, `rating_min`, and all cumulative match/goal counters are *recomputed at every snapshot*, so a 1950 row reflects only what was known by end-of-1950. This makes the dataset safe for backtesting — there is no information leakage from the future into any historical row.

## How Elo works for football

Elo is a paired-comparison rating system originally designed for chess. After each match, each team's rating updates by:

```
new_rating = old_rating + K * G * (W - W_e)
```

where `K` is a weight (higher for important matches), `G` adjusts for goal difference, `W` is the actual result (1 / 0.5 / 0), and `W_e` is the expected result given the pre-match rating gap and home advantage. The version used here (Lange's *World Football Elo Ratings*) has been published continuously since the early 1990s; the historical ratings have been backfilled to 1872 by retroactively applying the system to every recorded international.

Useful interpretation rules of thumb (from this rating system specifically):
- A **100-point gap** translates to roughly a **64%** win expectancy for the higher-rated team in a neutral-venue match.
- A **200-point gap** ≈ **76%** expectancy. **400-point gap** ≈ **91%**.
- Home advantage in this Elo variant is typically valued at about **+100 points**.

## Confederation and qualifying breakdown (48 teams)

| Confederation | Count | Teams |
|---|---|---|
| Hosts | 3 | United States, Canada, Mexico |
| UEFA | 16 | England, France, Croatia, Portugal, Norway, Germany, Netherlands, Switzerland, Scotland, Spain, Austria, Belgium, Bosnia and Herzegovina, Sweden, Turkey, Czechia |
| CONMEBOL | 6 | Argentina, Brazil, Ecuador, Paraguay, Uruguay, Colombia |
| CONCACAF (non-host) | 3 | Panama, Curaçao, Haiti |
| CAF | 9 | Morocco, Tunisia, Egypt, Algeria, Ghana, Cape Verde, Senegal, South Africa, Ivory Coast |
| AFC | 8 | Japan, Iran, Uzbekistan, Jordan, South Korea, Australia, Qatar, Saudi Arabia |
| OFC | 1 | New Zealand |
| Inter-confederation playoff | 2 | DR Congo, Iraq |

**Notable debutants (4):** Cape Verde, Curaçao, Jordan, Uzbekistan.
**Notable returners:** DR Congo & Haiti (first since 1974), Iraq (first since 1986).
**Notable absentee:** Italy (third straight World Cup missed; first time any former champion misses three in a row).

## Predecessor handling (history bridging)

Several of the 48 qualifiers exist today under a different name or political entity than they did historically. Where FIFA officially recognizes a successor relationship, this dataset folds the predecessor's rows into the modern team so you get one continuous time series. Six bridges were applied:

| Modern team | Predecessor(s) bridged | Predecessor era |
|---|---|---|
| Czechia (`CZ`) | Czechoslovakia (`CS`) | 1920–1993 |
| Germany (`DE`) | West Germany (`WG`) | 1949–1990 |
| Curaçao (`CW`) | Netherlands Antilles (`AN`) | 1924–2010 |
| DR Congo (`CD`) | Belgian Congo (`BC`), Congo-Léopoldville (`CJ`), Zaire (`ZR`) | 1948–1996 |
| Egypt (`EG`) | United Arab Republic (`RR`) | 1958–1961 |
| South Korea (`KR`) | pre-division Korea (`KA`) | 1942–1947 |

**East Germany (`DD`) is *not* bridged into Germany** — FIFA treats it as a separate national team that was dissolved, not a predecessor, which matches standard practice in the football statistics community.

## Provenance

- **Original source:** [World Football Elo Ratings](https://www.eloratings.net/) (Lange, eloratings.net).
- **Endpoints used:**
  - `https://www.eloratings.net/<YYYY>.tsv` — end-of-year snapshot for each year 1901–2025.
  - `https://www.eloratings.net/World.tsv` — current-day snapshot for the live row.
  - `https://www.eloratings.net/en.teams.tsv` — team code → human-readable name mapping.
- **Scrape method:** plain HTTPS GET; no auth, no rate-limit issues, no JavaScript execution required. The eloratings.net front-end is a thin SPA over these TSV files.
- **TSV column layout:** reverse-engineered from `https://www.eloratings.net/scripts/ratings.js` (the `datafield` definitions). The mid-row delta block (3 month / 6 month / 1y / 2y / 5y / 10y rank+rating changes) was *intentionally dropped* because its column count differs between the live `World.tsv` and the yearly `<YEAR>.tsv` files; you can reconstruct any delta by subtracting `rating[year=Y-k]` from `rating[year=Y]` for the same team.
- **Predecessor bridging:** custom post-processing layer applied in `scrape_history.py` based on FIFA-recognized federation lineage.

## Reproducibility

The scrapers used to build this file are:
- `scrape_elo.py` — single live snapshot, all 244 teams.
- `scrape_history.py` — full 1901–present history, all 244 teams.

Each runs in ~30s, requires only Python 3.10+ and `curl`, and writes a deterministic CSV. The filtering+bridging step that produces `elo_ratings_wc2026.csv` is included in the repository hosting these scripts (see the dataset description page on Kaggle).

## Update cadence

Planned re-scrapes:
1. **Once per week** through the end of pre-tournament friendlies (May–early June 2026).
2. **Immediately before kickoff** (June 10, 2026) — the canonical "pre-tournament" snapshot most useful for predictions.
3. **Daily during the tournament** so each match-day row reflects the prior day's results.
4. **Once at tournament close** (July 19, 2026) — final post-tournament ratings.

The `snapshot_date` column lets you find the as-of date of any row.

## Suggested uses

- **Pre-tournament predictions.** Filter to `snapshot_date == max(snapshot_date)` for each team and feed into your model. The 100-point ≈ 64% win-expectancy rule gives you a calibrated baseline.
- **Group-stage simulation.** Combine pre-tournament ratings with the published 2026 group draw to Monte-Carlo the group stage; the 48-team / 12-group / top-2 + best-8-third format is new for 2026, so reproducing it correctly is itself non-trivial.
- **Host advantage estimation.** `is_host == 1` is true for three teams. Combine with the historical "home" match counts (`matches_home / matches_total`) to test whether prior host-nation overperformance is captured by Elo alone or requires an additional dummy.
- **Era-controlled "all-time strength" rankings.** `rating_max` per team gives a quick read of historical peaks; `rating_avg` smooths out short bursts.
- **Predecessor analysis.** Compare a country's `rating_avg` in its predecessor era vs. its modern era — e.g. has Czechia underperformed or overperformed Czechoslovakia's mean?

## Caveats

- Elo treats friendlies and competitive matches differently (lower `K` weight for friendlies), but the *output ratings* here are already the post-update values — friendlies have already been down-weighted. You do **not** need to discount them further.
- Walkover and forfeit results are excluded from match counts upstream by eloratings.net.
- A handful of pre-1950 results are reconstructed from secondary sources rather than primary match reports; the upstream maintainer documents this on the methodology page.
- The Elo variant used here is *not* FIFA's official ranking (which uses a different formula since 2018). It is widely considered more predictive than the FIFA ranking; see Lasek et al. 2013, *International Journal of Forecasting* for an academic comparison.

## License & attribution

This dataset is published under **CC BY-SA 4.0**. If you use it, please credit:

- This Kaggle dataset (link on the dataset page), **and**
- The upstream source: **World Football Elo Ratings** (eloratings.net), maintained by Kirill Bukin and Erik Gebhardt continuing the work of Bob Runyan and the late Andrew Wallechinsky.

The upstream ratings themselves are published with a permissive reuse policy as documented on eloratings.net. The added value in this dataset is the time-series shape, the WC 2026 filtering, the predecessor bridging, and the column dictionary above.
