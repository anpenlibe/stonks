# Stonks

A playground for time-series modelling on Indian equities. The plan: pull a decade of price history for a diversified basket of NSE tickers, clean and align it, then use it as the substrate for a full pipeline of classical time-series and neural-network experiments. The project doubles as the submission for two university courses — one on regression / time-series, one on AI/ML — with course-specific notebooks under `rtsm/` and `aiml/`.

## Progress

- Pulled 10 years of daily data for the portfolio below into `data/raw/` via `scripts/fetch_data.py`.
- Aligned the individual series into a single modelling-ready frame via `scripts/align_prices.py`.
- **Sprint 01 — EDA done.** `scripts/01_eda.py` produces price history, log-return plot, return distribution, rolling stats, ACF/PACF, correlation heatmap, Q-Q plot, plus an ADF + KPSS stationarity table across all 7 tickers. Headlines: raw prices are non-stationary on both tests, log returns are stationary on both tests, RELIANCE excess kurtosis is ~9.2 (fat tails as expected), NSEI ↔ SENSEX correlation is 0.99. Full write-up in [`results/reports/01_eda.md`](results/reports/01_eda.md).
- **Sprint 02 — regime detection done.** `scripts/02_regime_detection.py` runs k-means on 60-day rolling vol/return features, plots an elbow curve across k=2..8, commits to k=2, and writes calm/turbulent labels to `data/processed/regime_labels.csv`. Headline: with k=2 the only stretch that separates as "turbulent" is COVID (2020-03-20 → 2020-06-30, 67 days, 2.8% of the labelled period). The 2022 rate-hike cycle does not surface as its own regime — COVID's volatility dwarfs everything else. Honest finding rather than a bug; flagged for downstream regime-split evaluation. Full write-up in [`results/reports/02_regime_detection.md`](results/reports/02_regime_detection.md).
- **Sprint 03 — preprocessing done.** `scripts/03_preprocessing.py` writes `data/processed/log_returns.csv` (2469 × 7), carves out a strictly time-based train/val/test split (1662 / 245 / 562 rows for 2016-04-12 → 2022-12-30 / 2023 / 2024-01-01 → 2026-04-10), fits per-ticker `MinMaxScaler(feature_range=(-1, 1))` on the training window only, persists scalers + scaled npz arrays under `data/processed/scaled/`, and exposes a reusable `make_windows(series, lookback=20)` helper for the MLP and LSTM scripts. Cross-tab confirms the Sprint 02 quirk: the entire turbulent regime sits in the training window — val and test are 100% calm. Full write-up in [`results/reports/03_preprocessing.md`](results/reports/03_preprocessing.md).
- **Sprint 04 — ARIMA/SARIMA done.** `scripts/04_arima.py` runs the full Box–Jenkins workflow on RELIANCE — manual order ID off the Sprint 01 ACF/PACF (`(1,0,1)`) cross-checked against `auto_arima` which picked `(0,0,0)` (i.e. constant drift + white noise — daily equity returns are nearly that pure), 2×2 residual diagnostics, walk-forward one-step-ahead forecast on the test window with prices back-transformed via `pred_price = actual_prev × exp(pred_log_return)`, and a SARIMA seasonality check at `m=5` that came in essentially tied (ΔAIC = +2.46, kept non-seasonal). Then summary-mode `auto_arima` + walk-forward for the other six tickers. Headlines: 5 of 7 tickers landed on `(0,0,0)`; SENSEX is `(1,0,0)`, HDFCBANK is `(2,0,2)`; directional accuracy spans 0.457 (TCS) to 0.525 (NSEI) — all within binomial noise of a coin flip on 562 trades. RELIANCE residuals fail Ljung-Box at lag 10/20 (p ≈ 0.007 / 0.0004) — leftover autocorrelation outside the `max_p=max_q=5` search radius and a clean motivation for GARCH (Sprint 05). Full write-up in [`results/reports/04_arima.md`](results/reports/04_arima.md).
- **Sprint 05 — GARCH(1,1) done.** `scripts/05_garch.py` fits GARCH(1,1) on RELIANCE in-sample log returns (Constant mean, Normal innovations) via the `arch` library, and runs a walk-forward one-step-ahead conditional-variance forecast on the test window using the GARCH recursion with parameters fixed at in-sample MLE. ARCH-effect tests are decisive: Ljung-Box on squared residuals at lag 20 gives p ≈ 1.8e-262, Engle's ARCH-LM gives p ≈ 2.8e-99 — the variance is unmistakably autocorrelated, exactly as the Sprint 04 Ljung-Box hinted. Fitted parameters (back-transformed to log-return scale): μ = +0.000877 (matches Sprint 04 ARIMA drift ✓), ω = 1.38e-05, α = 0.0793, β = 0.8734, **α + β = 0.9527** — persistent but short of IGARCH territory; vol-shock half-life ≈ 14 trading days. Test-period σ_t mean ≈ 0.0148 vs realised |return| mean ≈ 0.0099 — Normal-GARCH systematically underestimates extreme tails (max |return| 0.078 vs max σ 0.029), a clean future avenue for Student-t innovations. Full write-up in [`results/reports/05_garch.md`](results/reports/05_garch.md).
- **Sprint 06 — MLP done.** `scripts/06_mlp.py` trains a 4-layer feed-forward MLP (`20→64→32→16→1`, 3,969 params) per-ticker in PyTorch with Adam + MSE, max 100 epochs, batch 32, early stopping on val MSE with patience 10. Walk-forward 562 one-step-ahead test predictions match ARIMA's protocol exactly. CPU-pinned + seeded `DataLoader` makes reruns byte-deterministic. None of the 7 tickers hit MAX_EPOCHS=100 — RELIANCE early-stopped at epoch 14 with best val MSE = 0.006166 at epoch 4. Directional accuracy band 0.450 (ITC) → 0.523 (TCS), nearly identical to ARIMA's 0.457 → 0.525 (net average Δ vs ARIMA ≈ +0.01, all moves inside binomial noise at n=562). Honest finding: more capacity ≠ more skill on near-white-noise data; the MLP isn't beating ARIMA on this dataset, and that's the AI/ML-course-relevant lesson. Full write-up in [`results/reports/06_mlp.md`](results/reports/06_mlp.md).
- **Sprint 07 — LSTM done.** `scripts/07_lstm.py` trains two stacked LSTM layers in PyTorch (`LSTM(input_size, 64) → Dropout(0.2) → LSTM(64, 32) → last hidden state → Dropout(0.2) → Linear(32, 1)`, 29,729 params) per-ticker, with an optional multivariate variant on RELIANCE that feeds all 7 tickers' scaled log returns as parallel features (input_size=7, 31,265 params). Same Sprint 06 scaffolding (Adam lr=1e-3, MSE, max 100 epochs, batch 32, early stopping patience 10, walk-forward 562 one-step-ahead test predictions, CPU-pinned + generator-seeded DataLoader → byte-identical reruns confirmed). All 8 models early-stopped (range 11–34 epochs); RELIANCE univariate stopped at epoch 11 with best val MSE = 0.006162 at epoch 1. Directional accuracy band 0.457 (TCS) → 0.525 (NSEI), inside the same coin-flip noise band as ARIMA (0.457–0.525) and MLP (0.450–0.523); net Δ vs MLP ≈ −0.006 across 7 tickers — a wash. The multivariate variant gives Δ DA = +0.002 vs the univariate RELIANCE LSTM — a clean negative result for the "feeding NSEI/SENSEX in helps" hypothesis: contemporaneous corr 0.99 doesn't translate into next-day predictability. Full write-up in [`results/reports/07_lstm.md`](results/reports/07_lstm.md).
- **Sprint 08 — unified benchmarking done.** `scripts/08_benchmark.py` concatenates `arima_metrics.csv`, `mlp_metrics.csv`, `lstm_metrics.csv` into one long-form `master_benchmark.csv` (88 rows × 7 cols), reconstructs RELIANCE one-step-ahead test predictions for ARIMA / MLP / LSTM (per-script CSVs only saved aggregate metrics; the model classes from 04/06/07 are reloaded via `importlib` because their filenames start with digits), runs a Diebold–Mariano test on ARIMA vs LSTM, and writes three benchmark figures. Sanity-check: ARIMA RMSE re-computed in 08 matches `arima_metrics.csv` to 1e-12. Headlines: mean DA across 7 tickers — ARIMA 0.491, MLP 0.498, LSTM 0.492, multivariate-LSTM 0.511 (RELIANCE only); RELIANCE full-test RMSE band 0.01372 (multi-LSTM) → 0.01381 (MLP), all four models bunched within ~0.0007. **DM test: DM = 0.9803, p = 0.3269 → fail to reject H₀** that ARIMA and LSTM have equal squared-error forecast accuracy on RELIANCE — the LSTM's 1e-4 RMSE edge is well inside the variance of the loss differential. Full write-up in [`results/reports/08_benchmark.md`](results/reports/08_benchmark.md).
- **Sprint 09 — paper trader done.** `scripts/09_paper_trader.py` simulates BUY/HOLD/SELL on RELIANCE at a ±0.5% log-return threshold from ₹1,00,000 starting capital over the 562-day test window, against a buy-and-hold benchmark. Reuses Sprint 08's prediction-reconstruction helpers verbatim via `importlib`; trades execute at that day's Open price. Long-only, fractional shares, no transaction costs. **Headline: buy-and-hold wins decisively** — BuyAndHold +3.62% / Sharpe −0.089 / final ₹1,03,621; ARIMA 0.00% / 0 trades; LSTM 0.00% / 0 trades; MLP −8.47% / Sharpe −0.351 / 1 BUY on 2024-02-05 at ₹1,460.75 then held into the trough / final ₹91,528. Why the zero-trade outcomes are honest, not a bug: predicted log-return std is **5e-5 (LSTM)** and **0.00127 (MLP)** vs realised **0.01373** while the threshold is ±0.005 — the threshold filter is correctly rejecting un-actionable predictions. Same noise-floor story Sprints 04/06/07/08 told in RMSE/DA terms, now in P&L. Full write-up in [`results/reports/09_paper_trader.md`](results/reports/09_paper_trader.md).
- **Sprint 10 — sentiment extension done.** `scripts/10_sentiment.py` pulls RELIANCE.NS headlines via `yfinance.Ticker(...).news`, scores each one with VADER, aggregates to one row per IST publication date, smooths with a 5-day rolling mean, and aligns to next-day RELIANCE log returns for Pearson + Spearman. Cached headlines live in `data/processed/sentiment_raw.csv` so reruns are byte-stable; `SENTIMENT_REFRESH=1` forces a re-pull. **Honest outcome on this run: zero overlap.** yfinance returned 10 headlines spanning **2026-04-24 → 2026-05-06** — every sentiment date sits *after* the locked end-of-data (2026-04-10), so `align_to_next_day_returns` produces 0 pairs and the correlation is formally undefined (`n=0`). The optional sentiment-augmented LSTM is gated (need ≥ 100 aligned rows + ≥ 60 inside train+val) — closed cleanly without training on phantom data. VADER fires sensibly on the extremes; the middle band is noisy lexicon-on-financial-news, flagged. Full write-up in [`results/reports/10_sentiment.md`](results/reports/10_sentiment.md).
- **Course 1 notebook — `rtsm/notebook.ipynb`.** Eight-section narrative walk-through of the project from a Brockwell & Davis / Shumway & Stoffer angle (Introduction → Data & EDA → Market Regimes → ARIMA → GARCH → Regime-Split Evaluation → Paper Trading → Conclusion). 47 cells (26 markdown + 21 code), no re-modelling — every number is loaded from the existing `results/metrics/*.csv` and every plot embedded from `results/figures/`. Built deterministically by `scripts/build_rtsm_notebook.py` (stdlib `json` + `shutil`); re-running the builder produces a byte-stable notebook. `rtsm/figures/` mirrors the 16 figures the notebook embeds so the folder is a self-contained submission bundle.
- **Course 2 notebook — `aiml/notebook.ipynb`.** Nine-section narrative through the Bishop-PRML lens (Introduction → Data & Regime Detection via K-Means → Feature Engineering & Windowing → MLP with gradient descent / backprop → LSTM with gates + BPTT → Benchmarking → Sentiment Extension → Paper Trading → Conclusion). 40 cells (22 markdown + 18 code), built by the sibling `scripts/build_aiml_notebook.py` and self-contained under `aiml/` (15 figures mirrored into `aiml/figures/`). Spends most of its airtime on architecture decisions and training mechanics, then closes with the same honest finding: the neural nets did **not** beat ARIMA(0, 0, 0) on RMSE or DA, the Diebold–Mariano test confirms there's no real difference, and the paper trader translates that into buy-and-hold winning.

## The Portfolio

Chosen to span sectors with meaningfully different risk and cyclicality profiles, plus two broad-market benchmarks.

| Role | Name | Ticker | Why it's here |
| --- | --- | --- | --- |
| Benchmark | Nifty 50 | `^NSEI` | Overall health of the Indian market |
| Benchmark | Sensex | `^BSESN` | Broader BSE market sentiment |
| IT / Tech | Tata Consultancy Services | `TCS.NS` | Steady, secular growth |
| Banking / Finance | HDFC Bank | `HDFCBANK.NS` | Sensitive to rates and macro policy |
| Energy / Conglomerate | Reliance Industries | `RELIANCE.NS` | Massive market mover; highly traded |
| FMCG | ITC Limited | `ITC.NS` | Defensive; traditionally less volatile |
| Automotive | Maruti Suzuki | `MARUTI.NS` | Cyclical; sensitive to supply chains and consumer spending |

## Raw Data — `data/raw/`

One CSV per ticker, straight from yfinance. All series span **2016-04-11 → 2026-04-10**.

| File | Rows |
| --- | --- |
| NSEI.csv | 2464 |
| SENSEX.csv | 2462 |
| TCS.csv | 2470 |
| HDFCBANK.csv | 2470 |
| RELIANCE.csv | 2470 |
| ITC.csv | 2470 |
| MARUTI.csv | 2470 |

Indices have a handful fewer rows than equities — NSE and BSE observe a few extra holidays that still see equity trading.

## Aligned Data — `data/ARIMA/aligned_prices.csv`

Produced by `scripts/align_prices.py`. A single wide frame with every ticker's Adjusted Close on a common date index. Missing observations (holiday mismatches) are forward-filled from the previous day so every row is complete.

- **Index:** `Date`
- **Columns:** `HDFCBANK`, `ITC`, `MARUTI`, `NSEI`, `RELIANCE`, `SENSEX`, `TCS`

## Layout

```
Stonks/
├── data/
│   ├── raw/                 # per-ticker CSVs from yfinance (do not touch)
│   ├── ARIMA/               # aligned_prices.csv (do not touch)
│   └── processed/           # derived data
│       ├── log_returns.csv               # wide log-return frame, all 7 tickers
│       ├── regime_labels.csv             # k-means regime labels (calm / turbulent)
│       ├── arima_reliance_residuals.csv  # in-sample ARIMA residuals (Sprint 05's input)
│       ├── garch_volatility_forecast.csv # test-window σ_t one-step-ahead forecast
│       ├── sentiment_{raw,scores}.csv    # cached headlines + VADER scores
│       └── scaled/                       # per-ticker MinMaxScalers + scaled npz arrays
├── scripts/
│   ├── fetch_data.py             # pulls raw data
│   ├── align_prices.py           # builds aligned_prices.csv
│   ├── 01_eda.py                 # EDA: stationarity, distributions, ACF/PACF, correlations
│   ├── 02_regime_detection.py    # k-means regime detection, elbow curve, labels CSV
│   ├── 03_preprocessing.py       # log returns, time-based split, scalers, make_windows helper
│   ├── 04_arima.py               # ARIMA/SARIMA: Box–Jenkins on RELIANCE + summary mode for the rest
│   ├── 05_garch.py               # GARCH(1,1) on RELIANCE: ARCH-LM, fit, walk-forward variance forecast
│   ├── 06_mlp.py                 # PyTorch MLP 20→64→32→16→1: per-ticker, early stopping, walk-forward test
│   ├── 07_lstm.py                # Stacked LSTM (64→32) + Dropout: per-ticker univariate + RELIANCE multivariate
│   ├── 08_benchmark.py           # Master metrics concat + Diebold–Mariano test + benchmark figures
│   ├── 09_paper_trader.py        # BUY/HOLD/SELL @ open from ARIMA/MLP/LSTM preds vs buy-and-hold
│   ├── 10_sentiment.py           # VADER on yfinance news + correlation + augmented-LSTM gate
│   ├── build_rtsm_notebook.py    # deterministic builder for rtsm/notebook.ipynb
│   └── build_aiml_notebook.py    # deterministic builder for aiml/notebook.ipynb
├── results/
│   ├── figures/             # all PNG plots land here
│   ├── metrics/             # CSV summaries
│   ├── trades/              # paper-trading logs (one CSV per model)
│   └── reports/             # one markdown report per finished sprint
├── rtsm/                    # Course 1 deliverable (regression / time-series lens)
│   ├── notebook.ipynb
│   └── figures/             # mirrored from results/figures/ for self-containment
├── aiml/                    # Course 2 deliverable (AI/ML lens)
│   ├── notebook.ipynb
│   └── figures/
└── README.md
```

## How to reproduce

From the repo root, with `.venv` activated (or via `.venv/bin/python`):

```
python scripts/fetch_data.py             # one-off — only needed if data/raw/ is empty
python scripts/align_prices.py           # rebuilds data/ARIMA/aligned_prices.csv
python scripts/01_eda.py                 # writes results/figures/* and results/metrics/stationarity_tests.csv
python scripts/02_regime_detection.py    # writes data/processed/regime_labels.csv + regime figures
python scripts/03_preprocessing.py       # writes data/processed/log_returns.csv, scaled/, results/metrics/data_splits.csv
python scripts/04_arima.py               # writes arima_reliance_{diagnostics,forecast}.png, arima_metrics.csv, arima_reliance_residuals.csv
python scripts/05_garch.py               # writes garch_volatility.png, garch_params.csv, garch_volatility_forecast.csv
python scripts/06_mlp.py                 # writes mlp_{training_curve,reliance_forecast}.png, mlp_metrics.csv, mlp_reliance.pt
python scripts/07_lstm.py                # writes lstm_{training_curve,reliance_forecast}.png, lstm_metrics.csv, lstm_reliance.pt
python scripts/08_benchmark.py           # writes master_benchmark.csv, dm_test.csv, combined_forecast/regime_comparison/directional_accuracy.png
python scripts/09_paper_trader.py        # writes results/trades/*, paper_trading_summary.csv, equity_curves.png
python scripts/10_sentiment.py           # writes sentiment_{raw,scores}.csv, sentiment_correlation.csv, sentiment_{timeline,correlation}.png
python scripts/build_rtsm_notebook.py    # builds rtsm/notebook.ipynb (and copies referenced figures into rtsm/figures/)
python scripts/build_aiml_notebook.py    # builds aiml/notebook.ipynb (and copies referenced figures into aiml/figures/)
```

Dependencies: `yfinance pandas numpy matplotlib seaborn scipy statsmodels pmdarima scikit-learn torch vaderSentiment arch tqdm joblib`. Easiest path: `.venv/bin/pip install <those>` and you're good for the whole pipeline.
