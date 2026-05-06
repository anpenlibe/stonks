# Sprint 01 — Exploratory Data Analysis

## What got built

`scripts/01_eda.py` — a single, runnable script that loads `data/ARIMA/aligned_prices.csv`, computes log returns, and produces every plot + table the directive asked for. No models fit yet, this is purely "look at the data before doing anything to it".

Outputs:

- `results/figures/price_history.png` — all 7 tickers normalised to base 100, with placeholder regime guides at 2017-01-01 and 2020-01-01 (the actual regime boundaries get filled in once `02_regime_detection.py` runs).
- `results/figures/reliance_log_returns.png` — daily log returns for the hero ticker.
- `results/figures/reliance_return_distribution.png` — histogram + KDE + normal overlay.
- `results/figures/reliance_rolling_stats.png` — 60-day rolling mean and rolling std.
- `results/figures/reliance_acf_pacf.png` — ACF/PACF up to 40 lags.
- `results/figures/correlation_heatmap.png` — Pearson corr of log returns.
- `results/figures/reliance_qq_plot.png` — Normal Q-Q plot.
- `results/metrics/stationarity_tests.csv` — ADF + KPSS for raw prices and log returns across all 7 tickers.

## What the data looks like

- 2470 trading days, 2016-04-11 → 2026-04-10, 7 tickers, fully aligned.
- RELIANCE log returns: n=2469, mean ≈ 0.00072 (basically zero), std ≈ 0.0170.
- Most extreme days *both* land in March 2020: min log return -0.141 on 2020-03-23 (COVID lockdown crash), max +0.137 on 2020-03-25 (the rebound). That's a textbook volatility cluster — exactly the kind of thing GARCH is built for.

## Stationarity — exactly as the textbook predicts

Every raw price series fails ADF (p ≫ 0.05, so we cannot reject unit root) **and** fails KPSS (p = 0.01, so we reject stationarity). Both tests agree: raw prices are non-stationary. We knew this had to be true — equity prices don't have a fixed mean, they wander — but it's reassuring the tests confirm it cleanly.

Every log return series passes ADF (p between 1e-19 and 1e-28) and passes KPSS (p = 0.10). Both tests agree they're stationary. This is the green light to use ARIMA on log returns rather than levels in script 04. Reference: Brockwell & Davis Ch 3.

## Distribution — fat tails, as expected

Excess kurtosis on RELIANCE log returns is **9.22**. For a normal distribution this would be 0. So the empirical distribution has *much* heavier tails than the red dashed normal we overlaid. The Q-Q plot makes this obvious — both tails curve away from the reference line. Skew is small (0.14), so the distribution is roughly symmetric, just heavy-tailed.

This matters for two things later:
1. ARIMA assumes Gaussian innovations; we already know that's not quite true here, so we should expect the residual diagnostics in script 04 to flag it.
2. Volatility clustering is the *cause* of fat tails — script 05 (GARCH) is the right tool to model it.

## Cross-ticker structure

- NSEI ↔ SENSEX correlation = **0.99**. They're not literally the same index but, day-to-day in log returns, they basically are.
- RELIANCE ↔ NSEI = 0.65, RELIANCE ↔ SENSEX = 0.66. Makes sense — RELIANCE is one of the heaviest weights in both indices.
- HDFCBANK is the next most index-correlated single name (0.73 with NSEI, 0.75 with SENSEX) — financials carry a lot of weight in Indian indices.
- TCS is the most idiosyncratic name in the basket (0.47 with NSEI, 0.27 with RELIANCE). IT services seem to march to their own drummer compared to the rest.

This is a useful prior for the multivariate LSTM in script 07 — feeding NSEI/SENSEX in alongside RELIANCE has a real chance of helping, but TCS is unlikely to add much signal for a RELIANCE forecast.

## ACF / PACF on log returns

Both correlograms drop to near-zero almost immediately. There's a faint negative spike at lag 1 in both, suggesting maybe a very weak AR(1) or MA(1) structure, but nothing dramatic. This is consistent with the textbook view that daily equity returns are close to white noise in the *mean* — the predictability lives mostly in the *variance*, which is the GARCH story for script 05. The flat ACF/PACF will be a useful sanity check when `auto_arima` in script 04 picks a low-order ARIMA — it should, and if it picks something exotic that'd be a red flag.

## Dead ends / things tried

Nothing dropped this sprint — the directive's spec was concrete enough that I went straight from spec to implementation. Worth noting that I almost broke the placeholder regime lines into the regime-detection script, but the directive explicitly puts them in 01 with placeholder dates, so kept them where they belong.

## Sanity checks before moving on

- 14 stationarity rows in the CSV (7 tickers × 2 series). ✓
- 7 figures saved to `results/figures/`. ✓
- ADF/KPSS results match the expected pattern (non-stationary prices, stationary returns). ✓ — if any one ticker had defied this we'd have to chase it down before script 04.

## Queued for the next sprint

`scripts/02_regime_detection.py` — k-means on rolling vol/return features, elbow curve, k=2 fit, regime labels saved to `data/processed/regime_labels.csv`, and we re-render the price history with the *real* regime boundaries instead of the 2017/2020 placeholders.
