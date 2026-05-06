# Sprint 08 — Unified Benchmarking

## What got built

`scripts/08_benchmark.py` — the apples-to-apples comparison the previous six sprints have been pointing at. Loads `arima_metrics.csv`, `mlp_metrics.csv`, `lstm_metrics.csv`, stamps a `model` column on each, concatenates into one long-form `master_benchmark.csv`. Reconstructs RELIANCE one-step-ahead test predictions for ARIMA / MLP / LSTM (the per-script CSVs only saved aggregate metrics, but the Diebold–Mariano test needs aligned per-date errors), runs the DM test, writes three figures, and prints a narrative summary.

The reconstruction is RELIANCE-only and cheap: ARIMA(0,0,0) + 562-step `append(refit=False)` is a couple of seconds, and MLP / LSTM forward passes on the saved `.pt` state_dicts are instant on CPU. Saved `mlp_reliance.pt` and `lstm_reliance.pt` from Sprints 06/07 are reloaded via the original `MLP` / `LSTMRegressor` classes — imported through `importlib.util.spec_from_file_location` because the source filenames start with digits and plain `import` doesn't work. This reuse is intentional: no model code is duplicated, and the in-script ARIMA RMSE re-computation matches `arima_metrics.csv` to **1e-12** as a sanity check that the reconstruction is faithful.

## Outputs

- `results/metrics/master_benchmark.csv` — long form, **88 rows × 7 cols** (28 ARIMA + 28 MLP + 32 LSTM, including the multivariate row). Columns: `ticker, model, order, metric, full_test, regime_calm, regime_turbulent`. Schema-compatible with the per-script CSVs it concatenates from.
- `results/metrics/dm_test.csv` — one row, ARIMA vs LSTM on RELIANCE: `n=562, mean_loss_diff=2.58e-7, DM=0.9803, p=0.3269 → fail to reject H₀`.
- `results/figures/combined_forecast.png` — actual RELIANCE Adj Close + ARIMA / MLP / LSTM predicted prices on the 562-day test window. Multivariate LSTM excluded to keep the legend readable (would sit on top of the univariate line — Δ DA = +0.002 per Sprint 07).
- `results/figures/regime_comparison.png` — single-bar full-test RMSE per model on RELIANCE with a banner subtitle explaining why the directive's grouped calm-vs-turbulent design is degenerate on this test set.
- `results/figures/directional_accuracy.png` — horizontal bar chart of mean DA across 7 tickers per model + the multivariate-LSTM RELIANCE-only bar, with a 0.5 coin-flip line and a ±0.041 binomial-noise band shaded grey.

## Headline numbers

**Mean DA across 7 tickers (full test, n=562 per ticker):**

| model | mean DA |
|---|---|
| MLP | 0.498 |
| LSTM | 0.492 |
| ARIMA | 0.491 |
| LSTM_multivariate (RELIANCE only) | **0.511** |

The ±1.96σ binomial noise band around 0.5 at n=562 is **±0.041**. All four model means sit comfortably inside one noise band of 0.5 — no model has a real directional edge. The multivariate LSTM's 0.511 is the highest single number on the chart, but it's a single-ticker observation (RELIANCE only) inside the noise band, exactly as Sprint 07 already flagged.

**RELIANCE full-test RMSE band:**

| model | RMSE |
|---|---|
| LSTM_multivariate | 0.01372 |
| LSTM | 0.01374 |
| ARIMA | 0.01374 |
| MLP | 0.01381 |

All four bunch within **~0.0007** RMSE — the noise floor on daily log returns is doing the heavy lifting, not the model. ARIMA(0,0,0) and the univariate LSTM are byte-equivalent at five decimal places.

**Diebold–Mariano test (ARIMA vs LSTM, RELIANCE, h=1, squared-error loss):**
- n = 562
- mean(d_t) = 2.58e-7 (positive ⇒ ARIMA's MSE is microscopically larger)
- DM stat = **0.9803**
- p-value = **0.3269** (two-sided)
- **Verdict: fail to reject H₀.** ARIMA and LSTM forecast accuracy on RELIANCE squared-error loss are not statistically distinguishable on this test window.

The h=1 case is the easy one — the long-run-variance correction (Newey–West / Bartlett weights) only matters for h>1 because the loss differential `d_t = e_a² − e_b²` is uncorrelated at lag 0 ahead by construction. Plain sample variance suffices. Reference: Diebold & Mariano (1995), *Comparing Predictive Accuracy*, JBES.

## Reading the result

This is a **negative result with positive value**. The DM test isn't underpowered (n=562 is plenty for a one-step-ahead test) — it's correctly reporting that the LSTM's 0.0001 RMSE advantage on RELIANCE is well inside the variance of `d_t`. The narrative across Sprints 04–07 has been telegraphing this all along:

- Sprint 04 — `auto_arima` gives RELIANCE order **(0,0,0)**: AR/MA terms don't earn their AIC penalty.
- Sprint 06 — RELIANCE MLP early-stops at epoch **14** with best val MSE at **epoch 4**, then plateaus.
- Sprint 07 — RELIANCE LSTM early-stops at epoch **11** with best val MSE at **epoch 1**, multivariate Δ DA = +0.002.

When ARIMA(0,0,0) is the auto-search winner, when MLP and LSTM both find their best validation loss in the first handful of epochs, and when adding six correlated tickers as features moves DA by 0.2pp — the data is telling you, in three different formal idioms, that there is essentially no exploitable lag-structure in daily log returns. The DM p-value of 0.327 is the fourth confirmation. This is the "you-can't-beat-the-coin-flip" story the AI/ML and RTSM reports can lean on, with statistical evidence, not just intuition.

## Honest limitation: the regime comparison is degenerate

The directive's regime-comparison figure (grouped bars: calm RMSE vs turbulent RMSE per model) is **not possible** on this test set. Sprint 02's k=2 k-means labelling collapsed the entire turbulent regime to a single 67-day window in **2020-03-20 → 2020-06-30** (the COVID crash); everything after sits in the calm cluster. The training window contains all 67 turbulent days; the validation window is 100% calm; the test window (2024-01-01 → 2026-04-10) is 100% calm. Across every model's metrics CSV, `regime_turbulent` for the test period is therefore all-NaN.

Three options were considered in the plan:
1. Single-bar full-test RMSE per model + caveat banner.  ← **chosen**
2. Two-panel figure: test calm + train-window turbulent (slight apples-to-oranges since training data fit the model).
3. Drop the figure entirely.

Option 1 wins because (a) it doesn't pretend the directive's design is achievable on this dataset, (b) it doesn't smuggle in a half-fitted train-window comparison that would mislead a quick reader, and (c) the caveat in the subtitle directly cites Sprint 02 so a future reader can follow the trail. The figure still serves its grading purpose — same RMSE-per-model framing — without manufacturing data that isn't there.

A re-run with a different regime-detection scheme (rolling-volatility percentile bands; HMM with two states fit on a longer window) would put the 2022 rate-hike turbulence and the 2024–2026 stretches into the turbulent cluster and let the test-period regime split happen. That's a clean follow-up but explicitly out of scope for this sprint per the directive.

## Sanity checks

- ✓ `master_benchmark.csv` shape = **(88, 7)** = 28 + 28 + 32. No NaN-inducing column mismatch (verified by `pd.concat` with explicit shared `cols`).
- ✓ ARIMA RMSE re-computed in 08 = **0.013744623423** vs `arima_metrics.csv` = **0.013744623423** (Δ < 1e-12 — `walk_forward_forecast` is deterministic and the in-sample window matches Sprint 04's `split_train_test` exactly).
- ✓ MLP/LSTM state_dicts smoke-load via the original classes from `06_mlp.py` / `07_lstm.py` and produce 562 finite predictions on the saved `Xte`. Number of test windows matches the test date count exactly.
- ✓ DM test uses `ddof=1` sample variance (statsmodels convention) and a two-sided standard-normal p-value via `scipy.stats.norm.sf(|DM|) × 2`.
- ✓ Re-ran `python scripts/08_benchmark.py` twice, `master_benchmark.csv` and `dm_test.csv` MD5s identical (`md5sum` matches across runs). All upstream randomness is already pinned in 04/06/07; 08 only post-processes.
- ✓ `regime_comparison.png` plots all four models including LSTM_multivariate (the multivariate row is in `master_benchmark.csv`); `combined_forecast.png` deliberately excludes it.
- ✓ `directional_accuracy.png` mean-DA values verified by hand: ARIMA mean of 7 ticker DAs = (0.5249 + 0.5089 + 0.4573 + 0.4929 + 0.5089 + 0.4591 + 0.4858) / 7 = 0.4911. ✓
- ✓ DM `mean_loss_diff` is positive (+2.58e-7) — ARIMA's MSE is fractionally larger than LSTM's, consistent with `arima_rmse² > lstm_rmse²` in the per-ticker CSVs (0.0137446² > 0.0137352² → +2.58e-7 ✓).

## Dead ends / things tried

- **Newey–West weighted variance for the DM stat.** Tried it, then re-read Diebold & Mariano §3: at h=1 the loss-differential autocovariance kernel collapses to `γ_0` so plain sample variance is the right thing. Removed the kernel weights to keep the implementation faithful to the textbook formula. The numerical answer is the same up to floating-point noise either way.
- **Two-panel regime figure (test-calm + train-turbulent).** Drafted but dropped. Mixing test-window metrics with train-window metrics on the same chart conflates "model behaviour on held-out data" with "model behaviour on data it was fit on", which would be misleading at a glance. The Sprint 02 caveat banner is a more honest framing.
- **GARCH-informed variant in the master table.** The directive lists "(and the GARCH-informed variant if it materialises)". It didn't materialise — Sprint 05 produced volatility forecasts (`garch_volatility_forecast.csv`) but no mean-prediction model on top of them. Skipping it cleanly here keeps the master table to the four models we actually have.
- **Including the multivariate LSTM in the combined-forecast plot.** Dropped — the univariate and multivariate predicted-price lines sit on top of each other (RMSE Δ ≈ 0.0001), and the four-line figure already makes the noise-floor point. Added it as a bar in the DA chart instead, where the 0.511 vs 0.491–0.498 gap at least visually separates.
- **Per-ticker DM tests (six more, one per non-RELIANCE ticker).** The directive specifies ARIMA vs LSTM on RELIANCE specifically; given the cross-ticker DA bands are statistically indistinguishable from coin-flip already (Sprint 07's three-model head-to-head table), per-ticker DM would just be six more "fail to reject" rows. Skipped to keep the headline result clean.

## Queued for the next sprint

`scripts/09_paper_trader.py` — convert these one-step-ahead RELIANCE predictions into BUY / HOLD / SELL signals (threshold ±0.5%) and simulate from ₹1,00,000 starting capital against a buy-and-hold benchmark. Outputs: equity curves figure, per-trade log CSVs, a Sharpe-ratio summary table. Already-loaded RELIANCE preds from this script can be wired into 09 directly via the same `importlib` trick. The honest expectation, given the 0.491–0.511 DA band: none of the three models beats buy-and-hold consistently. That's worth confirming with real numbers.
