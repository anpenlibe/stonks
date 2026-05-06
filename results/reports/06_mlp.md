# Sprint 06 — Deep MLP via gradient descent + backpropagation

## What got built

`scripts/06_mlp.py` — first neural-net baseline. PyTorch MLP per directive: 20-day flat input → 64 ReLU → 32 ReLU → 16 ReLU → 1 linear (3,969 params), Adam (lr=1e-3) + MSE, max 100 epochs, batch 32, early stopping on val MSE with patience 10. One model per ticker; the RELIANCE state_dict is persisted for Sprint 09's paper-trader. Pinned to CPU explicitly — networks are tiny, training data is ~1640 rows × 7 tickers, and CPU + manual seeds + a generator-seeded `DataLoader` makes the run byte-deterministic across reruns.

The walk-forward protocol on the test set matches `04_arima.py` exactly: model trained once on (train, val) windows, then 562 one-step-ahead predictions on test using each window's prior 20 days as input. Test predictions come back in scaled space and are inverse-transformed via the train-only-fit `MinMaxScaler` from Sprint 03 before metrics. Realised log returns for scoring are loaded directly from `data/processed/log_returns.csv` so we don't round-trip through the scaler.

## Outputs

- `results/figures/mlp_training_curve.png` — RELIANCE train/val MSE over 14 epochs with the early-stopping epoch marked. The val curve is flat at ~0.006 from epoch 1, the train curve drops from 0.035 → 0.015 — a feature of train/val scale asymmetry, discussed below.
- `results/figures/mlp_reliance_forecast.png` — actual RELIANCE Adj Close vs MLP one-step-ahead forecast on the test window. Same overlap-by-design appearance as the ARIMA plot.
- `results/metrics/mlp_metrics.csv` — long form, 28 rows (7 tickers × 4 metrics), columns `ticker, order, metric, full_test, regime_calm, regime_turbulent`. The `order` column is reused as the architecture label `"20→64→32→16→1"` so this concats cleanly with `arima_metrics.csv` in Sprint 08's master table.
- `results/mlp_reliance.pt` — RELIANCE state_dict (20 KB). Sprint 09 will reload via `MLP(); model.load_state_dict(...)`.

## RELIANCE deep-dive findings

**Early stopping fires immediately.** Best val MSE = 0.006166 at **epoch 4**, full stop at epoch 14 (10-epoch patience expired). The model finds its noise floor inside the first handful of passes through the data and then there's nothing more to learn. This isn't a bug — it's a clean signal that the daily-log-return signal-to-noise ratio is genuinely low and a 3,969-param MLP saturates the predictable component fast.

**Train MSE > Val MSE — counter-intuitive but real.** The training curve shows the train loss (navy) sitting *above* the val loss (darkred) throughout. Normally val ≥ train, with overfitting widening the gap. Here it's the opposite — and it's not because the model is somehow "better on unseen data". It's because the train and val windows live on different volatility regimes:

- **Train (2016-04 → 2022-12)** contains the COVID 2020 stretch — RELIANCE log returns ranged from −0.144 to +0.137. After the train-only MinMaxScaler the train values fully fill `[−1, +1]`.
- **Val (2023)** is a calm year — scaled values land in `[−0.213, +0.317]`, well inside the unit interval.

A model predicting near-zero will produce near-zero MSE on val (small targets, small errors) but will rack up MSE on the COVID days in train regardless of how well the rest is fit. The train MSE that you see in the figure is dominated by ~67 high-magnitude COVID days that no flat-floor model is going to nail. Same dataset structure that made Sprint 03's leak-check inert (train and full-series extremes coincide) shows up here as a training-curve oddity. Documented loudly because a future reader looking at this plot will absolutely think "wait, val < train, is the dataloader leaking?" — it isn't.

**No overfitting signal.** Val MSE never starts climbing within the patience window for RELIANCE (or for any ticker). Across all 7 tickers, the worst case was NSEI which trained for 44 epochs before patience expired; everything else stopped within 24 epochs. Capacity isn't the bottleneck — the data is.

## Cross-ticker metrics

| ticker | RMSE | MAE | DA |
|---|---|---|---|
| TCS | 0.01366 | 0.00980 | **0.523** |
| NSEI | 0.00897 | 0.00635 | 0.512 |
| SENSEX | 0.00871 | 0.00618 | 0.505 |
| HDFCBANK | 0.01302 | 0.00915 | 0.505 |
| MARUTI | 0.01448 | 0.01043 | 0.502 |
| RELIANCE | 0.01381 | 0.00995 | 0.488 |
| ITC | 0.01212 | 0.00857 | 0.450 |

DA range **0.450 → 0.523** — almost identical to ARIMA's **0.457 → 0.525**. Both models live inside the same coin-flip-noise band, exactly as expected for daily log returns.

**MLP-vs-ARIMA head-to-head on directional accuracy:**

| ticker | ARIMA DA | MLP DA | Δ |
|---|---|---|---|
| TCS | 0.457 | 0.523 | **+0.066** |
| NSEI | 0.525 | 0.512 | −0.013 |
| SENSEX | 0.509 | 0.505 | −0.004 |
| HDFCBANK | 0.493 | 0.505 | +0.012 |
| MARUTI | 0.486 | 0.502 | +0.016 |
| RELIANCE | 0.509 | 0.488 | −0.021 |
| ITC | 0.459 | 0.450 | −0.009 |

Net: **MLP wins on 4 tickers, loses on 3, average Δ ≈ +0.01**. The TCS swing (+6.6pp) is the only really visible move; everything else is well inside the binomial noise band (≈ ±0.04 at n=562). Honest reading: the MLP isn't beating ARIMA on this data in any meaningful way. It's not losing either — it's matching a model with literally 2 parameters (μ and σ²) using a 3,969-param network. That's the AI/ML-course-relevant finding: more capacity ≠ more skill when the underlying signal is near-white-noise.

RMSE is essentially indistinguishable from ARIMA across the board (within ~1% on every ticker). Both models effectively predict "near zero" on log-return scale; one uses MLE on a normal mean, the other uses gradient descent on a non-linear regressor — they end up at the same place because that's where the data takes them.

**MAPE is junk again** (10⁹–10¹⁰), same divide-by-near-zero problem flagged in Sprint 04. Reported because the directive lists it; treat as non-informative.

**`regime_turbulent` is all NaN** — Sprint 02's k=2 labelling puts zero turbulent days in the test window. Carried through here exactly as in `arima_metrics.csv`. Resolution still deferred to Sprint 08.

## Sanity checks

- ✓ MLP parameter count = 3,969 = (20·64+64) + (64·32+32) + (32·16+16) + (16·1+1) = 1,344 + 2,080 + 528 + 17. Matches the architecture spec.
- ✓ All 7 tickers early-stopped (none hit `MAX_EPOCHS=100`); best-epoch range 2–34, full-stop range 12–44.
- ✓ `mlp_metrics.csv` shape (28, 6) — same as `arima_metrics.csv`. Schema-compatible for Sprint 08 concat.
- ✓ `regime_calm` matches `full_test` byte-for-byte (test is 100% calm).
- ✓ `mlp_reliance.pt` size = 20 KB, smoke-loaded back via `MLP(); model.load_state_dict(torch.load(...))` and a forward pass on `torch.zeros(1, 20)` returns a finite float.
- ✓ Train, val, test window counts are 1642 / 245 / 562 across all tickers — matches Sprint 03's split table exactly (with the `−LOOKBACK` offset for train).
- ✓ Predicted log-return distribution sits inside the realised distribution range (no exploded outputs from the inverse-transform).

## Dead ends / things tried

- **Considered training a single shared MLP across all tickers** (one model with input dim 20, trained on the union of all per-ticker windows). Skipped because the directive's metric schema is per-ticker and the scalers are per-ticker. Per-ticker models are also more comparable to ARIMA's per-ticker fits.
- **Considered `torch.use_deterministic_algorithms(True)` and the CUBLAS workspace config** for stronger reproducibility. Not needed once we pinned to CPU + seeded the train DataLoader's generator. Re-running gives identical numbers.
- **Considered larger lookback** (e.g. 60 days). Skipped — the directive specifies `lookback=20` for parity with the LSTM in Sprint 07. A wider window is something Sprint 08 might revisit if there's headroom; the early-stopping pattern here suggests the model isn't even using all 20 days meaningfully.
- **Considered a small dropout layer.** Skipped — the model isn't overfitting (val loss never climbs), so dropout would only slow training without helping.

## Queued for the next sprint

`scripts/07_lstm.py` — same training scaffolding as this script (train loop, early stopping, walk-forward, metrics CSV schema), but the input is reshaped to `(N, 20, 1)` and the architecture is two stacked LSTM layers (64 → dropout 0.2 → 32 → dropout 0.2) followed by a linear head. Plus an optional multivariate extension on RELIANCE that uses all 7 tickers' log returns as features (input dim 7 instead of 1). Training curves and forecast plot for RELIANCE; metrics for all 7 univariate models and the one multivariate model. Maps to "Deep Neural Networks, Backpropagation Through Time" from the AI/ML course — Bishop PRML Ch 5 plus the LSTM-gates intuition.
