# Sprint 02 — Market Regime Detection (k-means)

## What got built

`scripts/02_regime_detection.py` — runnable + importable, follows the same idioms as Sprint 01 (pathlib, `np.random.seed(42)`, seaborn `whitegrid`, conversational comments). It loads `data/ARIMA/aligned_prices.csv`, computes log returns, builds a 3-feature rolling-window matrix, runs k-means for k=2..8 (elbow curve), commits to k=2, relabels clusters by raw RELIANCE rolling vol so 0 = calm and 1 = turbulent, writes the labels CSV, and re-renders RELIANCE price with regime shading.

Outputs:

- `results/figures/regime_elbow_curve.png` — inertia vs k for k=2..8.
- `results/figures/regime_detection.png` — RELIANCE Adj Close with green (calm) / red (turbulent) shading.
- `data/processed/regime_labels.csv` — 2410 rows after the 60-day rolling-window warm-up. Columns: `Date`, `regime`, `regime_name`.

## Features used (per the directive)

Rolling 60-day window, three features:

- `reliance_vol` — std of RELIANCE log returns
- `reliance_mean` — mean of RELIANCE log returns
- `nsei_vol` — std of NSEI log returns (market-wide signal)

Standardised with `StandardScaler` before clustering — these features sit on different scales and k-means is distance-based, so without standardising the larger-magnitude feature would silently dominate the cluster geometry. Classic clustering gotcha and one of the first things class drilled in.

## Elbow curve

| k | inertia |
|---|---------|
| 2 | 3963.9 |
| 3 | 2642.7 |
| 4 | 1983.9 |
| 5 | 1587.0 |
| 6 | 1341.9 |
| 7 | 1161.1 |
| 8 | 1007.4 |

The curve doesn't have a sharp single elbow. The biggest drop is k=2 → k=3 (≈ 1320 reduction), and it keeps coasting down without an obvious knee. If we were free to pick k by elbow alone, k=3 would arguably be the more honest choice. But the directive frames the problem as calm vs turbulent (k=2), and that framing is what feeds downstream regime-split evaluation, so we stick with k=2 and report the trade-off here.

## What k=2 actually picks out

After remapping by mean raw vol:

- **Cluster 0 (calm):** mean reliance_vol = 0.0150 → 2343 days, **97.2%** of the labelled period.
- **Cluster 1 (turbulent):** mean reliance_vol = 0.0451 → 67 days, **2.8%** of the labelled period.

The single turbulent run in the entire 10-year window is **2020-03-20 → 2020-06-30** — the COVID lockdown crash and immediate aftermath. That's it. The 2022 rate-hike cycle that the directive expected to surface as turbulent does not separate from the calm cluster, because COVID's volatility (3-month rolling vol up to ~4.5%, vs 1.5% in calm periods) is in a different league from anything else in the sample. K-means with k=2 sees one extreme blob and a much-larger normal blob, and that's the partition you get.

This is a feature, not a bug, of letting the data decide. But it has a real consequence (see below).

## The downstream problem this creates

Test window in `03_preprocessing.py` is **2024-01-01 → 2026-04-10**. Under this regime labelling, every single test-window day is calm. So the original plan for `08_benchmark.py` — compute metrics on `regime_calm` and `regime_turbulent` test subsets and compare — is degenerate as currently specified: turbulent would be empty.

Two reasonable paths when we get to script 08:

1. Compute regime-conditional metrics across the *full* labelled period (training + val + test combined), accepting that "regime turbulent" effectively means COVID training-window performance. Honest but biased, since models are usually trained on this data.
2. Re-label using percentile bands on rolling vol (e.g. top 25% of rolling vol days = turbulent). This guarantees both regimes have presence in the test window. Drifts away from "let k-means decide" but produces a more useful evaluation split.

Punting the decision to script 08 with both options on the table.

## Sanity checks

- 2410 labelled rows = 2470 dates − 60-day rolling warm-up. ✓
- The single turbulent span lines up exactly with the COVID crash and rebound visible on the rolling-std plot from Sprint 01 (`results/figures/reliance_rolling_stats.png`). ✓
- 2 figures + 1 CSV produced; script runs end-to-end with no warnings except the usual benign sklearn deprecation chatter (none today). ✓

## Dead ends / things tried

Nothing was ripped out and re-tried this sprint. The first run produced the 97/3 split, and the temptation was to swap features around (drop `reliance_mean`, shrink the window, use percentile features) until k=2 split more "evenly". I held off — the project convention is to not quietly drift from the spec, and the spec itself frames this step as letting the data decide. So we let it.

The honest interpretation: the dataset really does have one extreme regime in 10 years. That's a finding worth reporting, not engineering around. The downstream consequences are documented; we'll deal with them in script 08.

## Queued for the next sprint

`scripts/03_preprocessing.py` — log-returns CSV, train/val/test split (2016-04-11 → 2022 / 2023 / 2024+), regime-based subsets pulled from this sprint's `regime_labels.csv`, per-ticker `MinMaxScaler` fit on the train set only (no leakage), scalers persisted with joblib to `data/processed/scaled/`, and a reusable `make_windows(series, lookback=20)` helper for the MLP and LSTM downstream.
