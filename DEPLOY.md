# Deploying the dashboard to Streamlit Community Cloud (free)

This hosts `app/dashboard.py` at a public `https://<you>-<repo>.streamlit.app` URL.

## What's already set up for you
- **`requirements.txt`** — slim runtime deps (the heavy pipeline libs are in
  `requirements-dev.txt`, not installed on the cloud).
- **`packages.txt`** — `libgomp1`, the system lib LightGBM needs on Linux.
- **`.streamlit/config.toml`** — dark theme.
- **`.gitignore`** — commits the small data the app needs (matches/features parquet,
  Elo ratings, models, results spine, shootouts, backtest report CSVs) and **excludes**
  the 92 MB FIFA CSV and the ~1.1 GB ESPN odds cache (both re-fetched / degraded at
  runtime, so the app still works without them).

## Steps

### 1. Put the repo on GitHub
From this folder (already a local git repo with an initial commit):

```bash
# create the GitHub repo and push (needs the GitHub CLI `gh`, logged in)
gh repo create soccer-model --private --source=. --remote=origin --push

# …or manually, if you made the repo in the GitHub web UI:
git remote add origin https://github.com/<you>/soccer-model.git
git branch -M main
git push -u origin main
```

(Private repo is fine — Streamlit Cloud can deploy from private repos on your account.)

### 2. Deploy on Streamlit Community Cloud
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → pick your `soccer-model` repo, branch `main`.
3. **Main file path:** `app/dashboard.py`
4. **Advanced settings → Python version: 3.12** (matches local; 3.11 also fine).
5. Click **Deploy**. First build takes a few minutes (installing LightGBM etc.).

### 3. (Optional) secrets
The app needs **no keys** to run — Kaggle/API-Football are optional and degrade
gracefully. If you ever add them, use **App → Settings → Secrets** (never commit them).

## Known limitations on the free tier (honest)
- **The bet Tracker resets on redeploy.** Streamlit Cloud's filesystem is ephemeral,
  so `clv_ledger.csv` / `clv_open.csv` don't persist across restarts. The tracker works
  within a session and re-syncs from ESPN, but it won't keep a permanent multi-week
  record. For a durable record you'd move to Render/Railway (persistent disk) or an
  external store — see the chat discussion.
- **No background scheduler.** The local daily snapshot task doesn't run on Cloud; the
  in-app "Auto-sync" (every few minutes while the page is open) covers live use.
- **Cold starts are a little slow** — the app refits Dixon-Coles + recomputes Elo on
  load (cached afterward). Fine on the free tier, just not instant on first hit.

## Updating the live site
Any `git push` to `main` auto-redeploys. To refresh the model's data (new results,
fresh ratings), run the pipeline locally, then commit the updated
`data/processed/*.parquet` + `elo_ratings.json` and push.
