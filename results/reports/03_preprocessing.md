# Sprint 03 — Preprocessing & Data Splits

## What got built

`scripts/03_preprocessing.py` — runnable + importable, same idioms as Sprints 01 and 02 (pathlib, `np.random.seed(42)`, conversational comments, `if __name__ == "__main__":` guard so the helpers can be imported without re-running the pipeline). It loads `data/ARIMA/aligned_prices.csv`, computes log returns, carves out the time-based train/val/test split, joins Sprint 02's regime labels, fits a per-ticker `MinMaxScaler` on the training window only, persists scalers + scaled arrays, exposes a reusable `make_windows(series, lookback=20)` helper for the MLP/LSTM scripts, and writes a splits-summary CSV that script 08 will consume.

Outputs:

- `data/processed/log_returns.csv` — wide log-return frame, 2469 rows × 7 tickers (one row dropped from the 2470-row aligned frame by the diff).
- `data/processed/scaled/{TICKER}_scaler.joblib` × 7 — sklearn `MinMaxScaler(feature_range=(-1, 1))` for each ticker.
- `data/processed/scaled/{TICKER}_scaled.npz` × 7 — npz with keys `train`, `val`, `test`, each a 1-D scaled log-return array.
- `results/metrics/data_splits.csv` — split summary with cross-tabulated regime counts (train_calm, train_turbulent, val_calm, …).

## Splits

| split | start | end | rows |
|---|---|---|---|
| train | 2016-04-12 | 2022-12-30 | 1662 |
| val   | 2023-01-02 | 2023-12-29 | 245 |
| test  | 2024-01-01 | 2026-04-10 | 562 |

`1662 + 245 + 562 = 2469` ✓ matches `log_returns.csv` row count, no overlap. Strictly chronological — no shuffling, no random sampling. Brockwell & Davis Ch 5 are blunt about why: shuffling a time series destroys the autocorrelation structure that's the whole point of the modelling, and lets the model peek at "future" information when scoring.

## Regime cross-tab

| split × regime | rows |
|---|---|
| train_calm | 1536 |
| train_turbulent | 67 |
| val_calm | 245 |
| val_turbulent | **0** |
| test_calm | 562 |
| test_turbulent | **0** |

This confirms the Sprint 02 finding cleanly: the entire turbulent regime under k=2 is the COVID window (2020-03-20 → 2020-06-30) and lives wholly inside the training set. Validation and test are 100% calm. Whatever script 08 ends up doing about regime-conditioned evaluation, the constraint is now visible in numbers.

## Scaling — why `(-1, 1)` and the leak-check awkwardness

Used `MinMaxScaler(feature_range=(-1, 1))` rather than the default `(0, 1)`. Log returns are signed and centred near zero, so a symmetric range preserves the directional signal. Default `(0, 1)` would map all losses to the lower half and all gains to the upper half but compress the centre; symmetric `(-1, 1)` keeps the sign-of-zero meaningful, which matters for the directional-accuracy metric the benchmark script eventually computes.

**Train-only fit is correctly enforced in code** — `fit_transform` is called only on the train slice, then `transform` (without re-fitting) on val and test. That's the canonical pattern Bishop PRML Ch 1 uses when introducing held-out evaluation.

The slightly awkward thing: I tried to do a numerical leak check (verifying the saved scaler's `data_min_`/`data_max_` differ from a "would-be-leaky" scaler fit on the full series). It came back identical for every ticker — but **not because of leakage**. It's because the COVID March 2020 crash is the global minimum for all 7 tickers and the COVID rebound is around the global maximum, and both sit inside the train window. So a scaler fit on train and a scaler fit on the full series produce literally the same `data_min_`/`data_max_`. The leak-check is undetectable on this dataset; the implementation is still correct (I re-read the code carefully). Worth noting because someone might re-run this sanity check and worry.

Scaled-range readout from the run:

| ticker | train min/max | val min/max | test min/max |
|---|---|---|---|
| NSEI     | −1.000 / +1.000 | +0.101 / +0.430 | −0.301 / +0.583 |
| SENSEX   | −1.000 / +1.000 | +0.107 / +0.422 | −0.278 / +0.584 |
| TCS      | −1.000 / +1.000 | −0.291 / +0.560 | −0.729 / +0.692 |
| HDFCBANK | −1.000 / +1.000 | −0.396 / +0.381 | −0.619 / +0.557 |
| RELIANCE | −1.000 / +1.000 | −0.213 / +0.317 | −0.546 / +0.501 |
| ITC      | −1.000 / +1.000 | −0.126 / +0.678 | −0.709 / +0.744 |
| MARUTI   | −1.000 / +1.000 | +0.026 / +0.413 | −0.114 / +0.725 |

Val and test stay strictly inside `[-1, +1]` for every ticker — as expected, since the train window contains the most extreme moves. If a future post-2022 day had exceeded the COVID-era range we'd see |scaled| > 1; that the values stay inside the unit interval is consistent with markets having been calmer in the 2023-2026 stretch than in 2016-2022.

## `make_windows` helper

Stride-tricks-based:

```python
windows = np.lib.stride_tricks.sliding_window_view(arr, lookback)
X = windows[:-1].copy()
y = arr[lookback:].copy()
```

Smoke-test on RELIANCE training returns: `X.shape = (1642, 20)`, `y.shape = (1642,)` — that's `1662 - 20`, exactly. Toy test: `make_windows(np.arange(25), lookback=20)` gives `X.shape == (5, 20)` and `y.shape == (5,)` with `y[0] == 20`. ✓

The LSTM script will reshape `X` to `(n, 20, 1)`; the MLP can use it flat.

## Sanity checks

- `1662 + 245 + 562 == 2469` ✓
- Date boundaries non-overlapping: `train.max() = 2022-12-30 < val.min() = 2023-01-02 < val.max() = 2023-12-29 < test.min() = 2024-01-01`. ✓
- All 7 scaler files + 7 npz files written under `data/processed/scaled/`. ✓
- `data_splits.csv` has 11 rows (3 base splits + 2 whole-period regime + 6 split×regime cross-tab). ✓
- Module is importable without side effects (the smoke-test print only runs under `__main__`). Verified via `importlib.util.spec_from_file_location`. ✓

## Dead ends / things tried

Nothing got ripped out and re-tried this sprint — the spec is mechanical enough that the first version ran cleanly. The one judgement call was the scaler `feature_range`: I considered `(0, 1)` for default-ness but rejected it because asymmetric scaling on signed returns is a small but real footgun for any later model that learns about sign. Symmetric `(-1, 1)` was the cleaner default. Documented in a code comment.

Considered persisting the regime-conditioned subsets as separate CSVs (e.g. `regime_calm_returns.csv`). Skipped it — `log_returns.csv` plus `regime_labels.csv` is fully sufficient, and the in-memory subset helper is exposed in `regime_subsets()` for any caller that wants it. Saving derived files that are trivial to reconstruct adds a maintenance burden without earning anything.

## Queued for the next sprint

`scripts/04_arima.py` — full ARIMA deep-dive on RELIANCE (manual order ID from the ACF/PACF Sprint 01 produced, auto_arima cross-check, residual diagnostics including Ljung-Box, rolling one-step-ahead forecast on the test window, SARIMA seasonality check at `m=5`), then summary-mode auto_arima + rolling forecast for the remaining six tickers. Metrics CSV with `full_test`, `regime_calm`, `regime_turbulent` columns — though per the table above, `regime_turbulent` will be empty in the test window, which is when the script-08 decision (training-window turbulent metrics vs percentile re-labelling) becomes unavoidable.
