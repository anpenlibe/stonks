# Sprint 04 — ARIMA / SARIMA Modelling

## What got built

`scripts/04_arima.py` — runnable + importable, same idioms as the earlier sprints (pathlib, `np.random.seed(42)`, conversational comments, `if __name__ == "__main__":` guard so the helpers can be reused without re-running the pipeline). It does the full Box–Jenkins workflow on RELIANCE — manual order ID off the Sprint 01 ACF/PACF, an `auto_arima` cross-check, residual diagnostics with Ljung-Box, walk-forward one-step-ahead forecast on the test window, and a SARIMA seasonality check at `m=5` — then runs summary-mode `auto_arima` + walk-forward for the other six tickers and saves a regime-sliced metrics CSV.

In-sample is **train+val combined** (1907 rows, 2016-04-12 → 2023-12-29). ARIMA has no early-stopping use for a held-out val set, so the canonical Box–Jenkins move is to estimate parameters on as much history as possible. Test stays identical to Sprints 03/06–08 (562 rows, 2024-01-01 → 2026-04-10) so the cross-model comparison in Sprint 08 stays apples-to-apples.

Walk-forward is implemented with `results.append(new_obs, refit=False)` rather than re-fitting at every step. True full re-estimation across 562 steps × 7 tickers would be wasteful and isn't what production rolling-forecast systems actually do (Hyndman's `forecast` package does the same trick). Coefficients are pinned at the in-sample MLE, the state filter just absorbs each new observation. Comment block in the script explains this design choice.

## Outputs

- `results/figures/arima_reliance_diagnostics.png` — 2×2 panel: residual time series (volatility clustering visible around step 1000, i.e. COVID), ACF of residuals (essentially flat by eye), histogram vs normal density (fat tails obvious), Q-Q plot (heavy tails departing from the line — exactly what you'd expect on financial returns and exactly what the EDA's excess-kurtosis ≈ 9.2 already told us).
- `results/figures/arima_reliance_forecast.png` — actual RELIANCE Adj Close vs ARIMA one-step-ahead forecast on the test window. Lines visually overlap because most of the "prediction" is yesterday's actual close × exp(small predicted log return) — covered honestly in the script and below.
- `results/metrics/arima_metrics.csv` — long form, 28 rows (7 tickers × 4 metrics) × 6 columns: `ticker`, `order`, `metric`, `full_test`, `regime_calm`, `regime_turbulent`.
- `data/processed/arima_reliance_residuals.csv` — 1907-row in-sample residual series, indexed by Date. This is Sprint 05's first input (ARCH-LM test + GARCH(1,1) fit).

## RELIANCE deep-dive findings

**Manual vs auto order.** Sprint 01's ACF/PACF for RELIANCE log returns is essentially flat — bars sit inside the confidence band at almost every lag, with maybe faint hints at lags 7, 10, and 11. My manual proposal was `(1, 0, 1)` — a small AR and MA term plus the EDA-confirmed `d=0`. `auto_arima` undercut me and picked **`(0, 0, 0)`**, AIC = −9951.71. That literally means *"return = constant drift + white noise"* — it's saying the AR/MA terms don't earn their AIC penalty. Honest reading: I was hedging with the (1,0,1), `auto_arima` was right that the data doesn't support it.

**Final fit.** ARIMA(0,0,0) on the 1907 in-sample rows. Drift coefficient = +0.0009 per day (≈ 0.09% mean return; statistically distinguishable from zero with p = 0.027), σ² = 0.0003. Statsmodels' built-in heteroskedasticity test in `model.summary()` reports H = 0.89 (p = 0.16) — borderline, doesn't reject homoskedasticity at 5% but residuals are visibly clustered when you plot them. The Jarque-Bera p-value is < 1e-300 — residuals are decisively non-normal, with kurtosis 12.33. None of this is surprising. It's all the standard "daily equity returns" signature.

**Ljung-Box on residuals — a tension worth flagging.** At lag 10 the test gives stat = 24.28, p = 0.0069. At lag 20: stat = 48.08, p = 0.0004. Both reject the white-noise null at 5%. So residuals *aren't* clean white noise — there's some autocorrelation `auto_arima` didn't capture. The reason becomes obvious if you stare at it: ARIMA(0,0,0) literally just subtracts the mean, so residuals = returns − mean. Whatever lag structure the returns have, the residuals inherit. The ACF in Sprint 01 had touches at lags 7, 10, 11, 18 — outside the `max_p = max_q = 5` search radius `auto_arima` was given. A wider search might have picked up an MA(7) or AR(10) term, but I'm honouring `auto_arima`'s choice rather than re-tuning until residuals look pretty. Real takeaway for the report: ARIMA on log returns leaves variance structure on the table, which is exactly what GARCH (Sprint 05) is built for.

**Forecast plot reading.** The actual price and the predicted price overlap so closely you can barely tell them apart at the default zoom. This is *not* "the model works great" — it's a structural feature of one-step-ahead price plots. Predicted price[t] = actual price[t-1] × exp(small predicted log return), and the predicted log return is tiny, so what you're really plotting is "yesterday's price ≈ today's price". Useful as a sanity check (the back-transform isn't exploding, no off-by-one errors in date alignment) but not as a measure of forecast quality. Use the metrics CSV for that.

**SARIMA at m=5.** `auto_arima` with `seasonal=True, m=5` picks ARIMA(0,0,0)(2,0,0,5) — i.e. seasonal AR terms at lags 5 and 10 stacked on top of the non-seasonal (0,0,0). AIC = −9954.17. Compared to the non-seasonal model's −9951.71, that's a delta of just +2.46 — below the rule-of-thumb threshold of ~5 for a "material" AIC improvement. The seasonal AR is plausibly catching the lag-5/lag-10 ACF bumps that the Ljung-Box was complaining about, but the improvement is marginal. Per directive, keep the non-seasonal model. Comment in the script uses the directive's exact phrasing: *"tried adding a seasonal component with m=5 (weekly), but it didn't improve AIC meaningfully — makes sense, daily stock returns don't have strong weekly seasonality."*

## Cross-ticker metrics

| ticker | order | RMSE (full_test) | MAE | DA |
|---|---|---|---|---|
| NSEI | (0,0,0) | 0.00875 | 0.00616 | **0.525** |
| SENSEX | (1,0,0) | 0.00869 | 0.00613 | 0.509 |
| RELIANCE | (0,0,0) | 0.01374 | 0.00992 | 0.509 |
| HDFCBANK | (2,0,2) | 0.01306 | 0.00919 | 0.493 |
| MARUTI | (0,0,0) | 0.01429 | 0.01036 | 0.486 |
| ITC | (0,0,0) | 0.01205 | 0.00849 | 0.459 |
| TCS | (0,0,0) | 0.01365 | 0.00984 | 0.457 |

Five of seven tickers got ARIMA(0,0,0) — the same "constant drift + white noise" story as RELIANCE. Only **SENSEX** (AR(1)) and **HDFCBANK** (ARIMA(2,0,2)) had non-trivial linear structure that `auto_arima` thought was worth the AIC penalty. HDFCBANK's (2,0,2) is interesting; it's the only ticker where AR and MA both contribute, possibly reflecting bank-specific autocorrelation in the 2018-2020 window (when HDFCBANK had its own private corrections separate from the broad market).

**Directional accuracy band: 0.457 → 0.525.** Hovering on either side of the coin-flip line, exactly as expected for ARIMA on daily equity returns — Brockwell & Davis Ch 5 effectively warn that mean-reverting linear models on near-white-noise series can't beat random sign-prediction by much. NSEI scrapes 52.5% which would beat a coin flip if the gap is real; on 562 trades that's still inside binomial noise (95% CI for p=0.5 is roughly ±0.04). Honest: ARIMA isn't going to make anyone rich.

**RMSE pattern.** Index returns (NSEI, SENSEX) have noticeably lower RMSE (~0.0087) than individual stocks (~0.012–0.014). That's the diversification effect — a basket's daily move has lower variance than any single constituent, so the "constant mean" prediction is closer on average. Doesn't say anything about which ticker has the more *predictable* return; just about which has the smaller scale.

**MAPE is unusable.** Values come out at 10⁹–10¹⁰. This isn't a bug — it's MAPE doing what MAPE always does on data that crosses zero. Daily log returns can be 10⁻⁶ or smaller in magnitude, so dividing the absolute error by `|y_true|` blows up. The script computes it for completeness because the directive lists it as one of the four metrics, and a comment in `compute_metrics` flags this explicitly. For Sprint 08's benchmark, treat RMSE/MAE/DA as the headline numbers and ignore the MAPE column.

**`regime_turbulent` is all NaN.** Sprint 02's k=2 labelling puts zero turbulent days in the test window (the entire turbulent regime is the COVID 2020 stretch, which sits in the training set). So the regime-turbulent metrics column has nothing to average over → NaN. This was forecast in the Sprint 02 and Sprint 03 reports and the directive flags it as a Sprint 08 problem. Surfacing it in the CSV honestly rather than deleting the column means Sprint 08 sees the gap immediately.

`regime_calm` matches `full_test` byte-for-byte — same reason: test is 100% calm.

## Sanity checks

- ✓ `arima_metrics.csv` shape = (28, 6); 7 tickers × 4 metrics.
- ✓ Residual CSV has 1907 rows = in-sample size.
- ✓ Both figures saved at 150 DPI, render correctly when opened.
- ✓ Walk-forward predictions length matches test rows (562) for every ticker.
- ✓ `regime_calm` and `full_test` columns are identical (test is 100% calm — they should be).
- ✓ `regime_turbulent` is NaN for every row (test has zero turbulent days).
- ✓ Drift coefficient (+0.0009) is positive — RELIANCE went up on average over the in-sample window, which matches the price chart.

## Dead ends / things tried

- **Manual order (1,0,1) lost to auto_arima's (0,0,0).** Not really a dead end — it's the directive's intended workflow. Documented honestly in the script with the directive's exact-phrasing comment.
- **SARIMA(0,0,0)(2,0,0,5) was a near-tie.** AIC delta = +2.46, below the cutoff. Kept non-seasonal. If the threshold were stricter the SARIMA would win on a technicality, but the BIC penalty for seasonal terms (4 extra params, each fitted to seasonal blocks) makes the case weaker still. Real point: weekly seasonality on daily Indian equity returns is a marginal signal at best.
- **Ljung-Box tension left in.** I considered widening `auto_arima`'s `max_p`/`max_q` to 10 to chase the lag-7/lag-10 ACF bumps, but that veers into post-hoc tuning. Honest version: ARIMA(0,0,0) leaves the lag-10 autocorrelation in the residuals; that's a real limitation of mean-only models on financial returns; GARCH (Sprint 05) and the neural nets (06/07) are how we earn that signal back.
- **MAPE is junk on log returns.** Considered replacing with sMAPE or omitting the column. Kept it as-is because the directive lists it explicitly — the report flags it as the report should.

## Queued for the next sprint

`scripts/05_garch.py` — GARCH(1,1) on RELIANCE. Specifically:

- Load the residuals saved here at `data/processed/arima_reliance_residuals.csv`.
- Engle's ARCH-LM test on residuals (Ljung-Box on **squared** residuals — tests for autocorrelation in the variance, not the mean). Given the visible volatility clustering in the residual time-series plot above and the kurtosis = 12.33, the ARCH effect is nearly guaranteed to be there.
- Fit GARCH(1,1) via the `arch` library on the in-sample log returns (or on residuals, depending on how the spec reads — the directive's wording leans towards returns directly).
- Plot conditional volatility σ_t over the full period with regime boundaries annotated, and overlay |log returns|.
- Rolling one-step-ahead variance forecast on the test window using the same walk-forward protocol as ARIMA.
- Persist α, β, ω, persistence = α+β to `results/metrics/garch_params.csv`. Expectation per textbooks: persistence very close to 1 — equity volatility is sticky.
