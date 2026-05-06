# Sprint 07 — LSTM via Backpropagation Through Time

## What got built

`scripts/07_lstm.py` — the flagship sequential model. Two stacked LSTM layers in PyTorch with explicit dropout between them: `LSTM(input_size, 64) → Dropout(0.2) → LSTM(64, 32) → take last hidden state → Dropout(0.2) → Linear(32, 1)`. Trained per-ticker with Adam (lr=1e-3) + MSE, max 100 epochs, batch 32, early stopping on val MSE with patience 10. Same scaffolding as Sprint 06 — the train loop, the early-stopping logic, the walk-forward test protocol, the regime-sliced metrics builder, the deterministic seed plumbing — copied verbatim so the comparison stays apples-to-apples.

The directive's optional multivariate variant is included on RELIANCE: same architecture but `input_size=7`, fed all 7 tickers' scaled log returns as parallel features at every timestep. Same scaler-based inverse-transform on the predictions (RELIANCE scaler, since RELIANCE is column 0 of the feature stack and the y-target).

Walk-forward 562 one-step-ahead test predictions per model, identical date index to ARIMA / MLP. CPU-pinned + generator-seeded `DataLoader`; verified byte-identical reruns.

## Outputs

- `results/figures/lstm_training_curve.png` — RELIANCE univariate train/val MSE over 11 epochs with the early-stopping epoch marked.
- `results/figures/lstm_reliance_forecast.png` — actual RELIANCE Adj Close vs LSTM one-step-ahead forecast on the test window. Same overlap-by-design appearance as the ARIMA / MLP plots (predicted price = previous actual close × exp(predicted log return) — close lags by one day so the lines hug each other).
- `results/metrics/lstm_metrics.csv` — long form, **32 rows × 6 cols**: 7 univariate tickers × 4 metrics + 1 multivariate × 4 metrics. Schema-compatible with `arima_metrics.csv` and `mlp_metrics.csv` for Sprint 08's master-table concat. Multivariate row uses `order = "LSTM_multivariate"` to disambiguate from the univariate RELIANCE row.
- `results/lstm_reliance.pt` — RELIANCE univariate state_dict (123 KB). Sprint 09's paper-trader will reload via `LSTMRegressor(input_size=1); model.load_state_dict(...)`.

## RELIANCE deep-dive findings

**Param counts:** univariate LSTM = **29,729** params, multivariate = **31,265** params. The MLP from Sprint 06 was 3,969 params — the LSTM is ~7.5× larger by parameter count, almost all of it inside the gate matrices of `lstm1`. Sanity-check: LSTM1 (input=1, hidden=64) has 4·(1·64 + 64·64 + 2·64) = 17,152 params; LSTM2 (input=64, hidden=32) has 4·(64·32 + 32·32 + 2·32) = 12,544; head 33; total 29,729. ✓

**Early stopping fires almost as fast as the MLP.** RELIANCE univariate stopped at epoch **11**, with best val MSE = **0.006162** at epoch **1** — same noise-floor pattern. Across the 7 univariate models the stop epoch range was 11–34 (TCS the slowest); none came close to MAX_EPOCHS=100. The multivariate model went 20 epochs (best at epoch 10). The signal-to-noise on daily log returns is genuinely too low for any of these to keep finding structure.

**Train MSE > Val MSE again.** Same story as MLP: train contains the COVID 2020 stretch (scaled to ±1), val is the calm 2023 stretch (scaled to roughly [−0.21, +0.32]), and the train MSE is dominated by the high-magnitude COVID days the model can't nail. Documented loudly in Sprint 06's report; carries over here unchanged because the scalers and split are unchanged.

**The LSTM does not exploit temporal structure on this dataset.** RELIANCE univariate LSTM converges to essentially the same operating point as the MLP — best val MSE 0.006162 (LSTM) vs 0.006166 (MLP), test RMSE 0.01374 (LSTM) vs 0.01381 (MLP), test DA 0.509 (LSTM) vs 0.488 (MLP). Two stacked gated recurrent layers find essentially the same flat-floor predictor as a 4-layer feed-forward net. Honest read: the gates aren't earning their parameters, because there is no reliable lag-structure in the daily log returns to gate in the first place. (Sprint 04's ARIMA `(0,0,0)` agrees from the classical side: AR/MA terms didn't earn their AIC penalty either.)

## Cross-ticker metrics (univariate)

| ticker | RMSE | MAE | DA |
|---|---|---|---|
| NSEI | 0.00876 | 0.00616 | **0.525** |
| SENSEX | 0.00871 | 0.00614 | 0.509 |
| RELIANCE | 0.01374 | 0.00992 | 0.509 |
| HDFCBANK | 0.01298 | 0.00913 | 0.500 |
| MARUTI | 0.01429 | 0.01036 | 0.486 |
| ITC | 0.01220 | 0.00868 | 0.459 |
| TCS | 0.01368 | 0.00988 | 0.457 |

DA range **0.457 → 0.525** — sits inside the same coin-flip-noise band as ARIMA (0.457 → 0.525) and MLP (0.450 → 0.523). At n=562 the binomial noise band around 0.5 is roughly ±0.04, so every single ticker is statistically indistinguishable from random direction-guessing.

## Three-model head-to-head DA

| ticker | ARIMA | MLP | LSTM_uni | Δ(LSTM−MLP) |
|---|---|---|---|---|
| NSEI | 0.525 | 0.512 | **0.525** | +0.012 |
| SENSEX | 0.509 | 0.505 | 0.509 | +0.004 |
| TCS | 0.457 | **0.523** | 0.457 | **−0.066** |
| HDFCBANK | 0.493 | 0.505 | 0.500 | −0.005 |
| RELIANCE | 0.509 | 0.488 | **0.509** | +0.021 |
| ITC | 0.459 | 0.450 | 0.459 | +0.009 |
| MARUTI | 0.486 | 0.502 | 0.486 | −0.016 |

Net: **LSTM wins on 4 tickers, loses on 3** vs MLP, average Δ = −0.006 (essentially zero). The TCS swing (−6.6pp) is the only visible move and goes the other way from the MLP's TCS gain in Sprint 06 — neither the LSTM nor the MLP has a real edge there; both are just landing on different sides of a coin-flip distribution. Notably the LSTM matches ARIMA's DA byte-for-byte on five of seven tickers (NSEI, SENSEX, TCS, RELIANCE, ITC) — a strong hint that all three models are converging on the "predict near zero" strategy and the DA = sign-coincidence rate is essentially being read off the realised return signs alone.

RMSE differences across models on each ticker are all <2% — the three live in the same neighbourhood by every regression metric. This is the AI/ML-course-relevant lesson, restated: when the data is near-white-noise, model capacity doesn't translate into skill.

## Multivariate vs univariate (the interesting bit)

| variant | RMSE | MAE | DA |
|---|---|---|---|
| LSTM univariate (RELIANCE only) | 0.01374 | 0.00992 | 0.509 |
| LSTM multivariate (all 7 tickers) | 0.01372 | 0.00991 | **0.511** |
| Δ | −0.00001 | −0.00001 | +0.002 |

The hypothesis was that NSEI ↔ SENSEX correlations of 0.99 and a broader sectoral signal across the 7 tickers would give the multivariate LSTM something to work with. The honest answer: **no measurable lift**. Δ DA = +0.2pp at n=562 is well inside binomial noise (±4pp band). Best val MSE: 0.006157 (multi) vs 0.006162 (uni) — the multivariate model finds a microscopically better local minimum, but the test-set transfer is essentially the same. The directive expected this might be the place where a real lift shows up — it didn't.

Two readings:
- **Pessimist:** the contemporaneous correlation across tickers is real but it doesn't translate into next-day predictability. If NSEI is up at time `t`, RELIANCE is also up at time `t` (corr 0.99), but knowing NSEI's *previous 20 days* doesn't help predict RELIANCE's *next day* any better than RELIANCE's own previous 20 days.
- **Architectural:** the gate matrices grow only modestly (input_size 1 → 7 adds 4·6·64 = 1,536 params to lstm1) and may not have the headroom to learn cross-ticker dependencies at this depth. A wider first hidden state, or a CNN-style feature extractor before the LSTM, might do more — out of scope for Sprint 07.

Either way, this is a clean negative result rather than a wash, and it's the kind of finding the AI/ML report can lean on: more features ≠ more skill on near-white-noise targets.

## Sanity checks

- ✓ Univariate LSTM param count = 29,729 = 17,152 (lstm1) + 12,544 (lstm2) + 33 (head). Matches the architecture spec.
- ✓ Multivariate input_size = 7, lstm1 widens to 18,688 params, total 31,265.
- ✓ All 8 models early-stopped (none hit MAX_EPOCHS=100); stop range 11–34, best-epoch range 1–24.
- ✓ `lstm_metrics.csv` shape (32, 6) — schema columns identical to `mlp_metrics.csv`. Concat with `arima_metrics.csv` + `mlp_metrics.csv` gives 88 rows with no NaN-inducing column mismatch.
- ✓ `regime_calm` matches `full_test` byte-for-byte across all 32 rows (test window is 100% calm, same as MLP/ARIMA).
- ✓ `regime_turbulent` is all-NaN — Sprint 02's k=2 labelling, unchanged. Resolution still deferred to Sprint 08.
- ✓ Determinism: ran twice, `lstm_metrics.csv` is byte-identical (`diff` empty) — CPU-pinned + seeded numpy/torch + generator-seeded DataLoader holds.
- ✓ `lstm_reliance.pt` size = 123 KB, smoke-loaded back via `LSTMRegressor(input_size=1); model.load_state_dict(torch.load(..., weights_only=True))` and a forward pass on `torch.zeros(1, 20, 1)` returns a finite scalar.
- ✓ Train / val / test window counts are 1642 / 245 / 562 across all univariate models, and 1642 / 245 / 562 for the multivariate variant — same alignment as MLP.
- ✓ Multivariate windows: each per-ticker `train` array has length 1662 (so windowed length 1642 ✓); per-split lengths agree across all 7 tickers (`assert` on the stack guards this in code).
- ✓ Multivariate y-target column verified: `feature_order = [HERO] + [t for t in TICKERS if t != HERO]` puts RELIANCE at column 0, and `mat[LOOKBACK:, 0]` is what gets returned as y. Inverse-transformed with the RELIANCE scaler.

## Dead ends / things tried

- **`nn.LSTM(num_layers=2)` instead of two explicit LSTM layers.** Dropped because PyTorch's built-in inter-layer dropout applies to the inputs of layers 2..N — close in spirit to the directive but the wiring is less obvious to a reader. The explicit `lstm1 → drop1 → lstm2 → drop2 → head` form makes the directive's "return_sequences=True → dropout → next LSTM" exact.
- **`torch.use_deterministic_algorithms(True)`.** Not needed once we pinned to CPU and seeded the train DataLoader's generator. Re-running gives byte-identical metrics.
- **A wider lookback (60 days).** Skipped — directive specifies 20, and the early-stopping pattern (best epoch ≤ 10 on most tickers) suggests the model isn't even using all 20 days meaningfully.
- **Bidirectional LSTM.** Bidirectional doesn't make sense for one-step-ahead forecasting — it would peek at the future during the forward sequence pass. Not tried.
- **Heavier multivariate features (volume, volatility from Sprint 05).** Tempting but out of scope for Sprint 07's spec; the directive frames this as the "all 7 log returns as features" experiment specifically. Worth revisiting if Sprint 08's master table flags the multivariate row as a contender.

## Queued for the next sprint

`scripts/08_benchmark.py` — unified benchmarking. Concats `arima_metrics.csv`, `mlp_metrics.csv`, `lstm_metrics.csv` (and the GARCH-informed variant) into one master comparison table; computes Diebold-Mariano significance between ARIMA and LSTM on RELIANCE; writes a final figures + summary CSV. The Sprint 02 regime-split caveat is finally going to bite here — the test window is 100% calm, so the regime breakdown collapses to "full test" for every model. The benchmark report needs to surface that loudly rather than silently dropping the turbulent column.
