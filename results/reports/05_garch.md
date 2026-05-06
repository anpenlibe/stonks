# Sprint 05 — GARCH Volatility Modelling

## What got built

`scripts/05_garch.py` — the variance counterpart to Sprint 04's mean model. ARIMA(0,0,0) on RELIANCE essentially boiled down to "tomorrow's return ≈ small constant drift + white noise", and the Ljung-Box on residuals failed at lag 10 / 20. A lot of that "leftover autocorrelation" is actually variance clustering — calm days follow calm days, turbulent days follow turbulent days — which is exactly the structure GARCH is built to capture (Brockwell & Davis Ch 4).

The script (a) runs two complementary ARCH-effect tests on the residuals saved in Sprint 04, (b) fits GARCH(1,1) on the in-sample log returns via the `arch` library with a constant mean and normal innovations, (c) does a walk-forward one-step-ahead conditional-variance forecast on the test window using the GARCH recursion `σ²_{t+1} = ω + α·ε²_t + β·σ²_t` with parameters fixed at in-sample MLE (same `refit=False` trick as Sprint 04), and (d) plots σ_t over the full period with the COVID regime shaded.

In-sample = train+val combined (same as Sprint 04, 1907 rows). Returns are scaled ×100 for the optimiser's numerical stability — that's the `arch` library's documented recommendation — and then back-transformed for the saved σ values and parameter CSV so everything stays on log-return scale.

## Outputs

- `results/figures/garch_volatility.png` — single panel: |log return| in light grey as the backdrop, in-sample σ_t (navy), test-period one-step-ahead σ_t (darkred dashed), train/test boundary marker, COVID turbulent regime shaded in red. The σ_t envelope hugs the |return| backdrop tightly on calm days and spikes to ~0.07–0.08 during the shaded COVID stretch — visually unambiguous variance clustering.
- `results/metrics/garch_params.csv` — long form, 8 rows: μ, ω, α, β, persistence, log-likelihood, AIC, BIC.
- `data/processed/garch_volatility_forecast.csv` — 562-row test-period σ_t series, indexed by Date. Sprint 08 will use this for regime-conditioned variance evaluation.

## ARCH-effect tests — decisive

Both tests reject "no ARCH" by absurd margins.

**Ljung-Box on squared residuals.** Tests autocorrelation in the variance, not the mean.

| lag | LB stat | p-value |
|---|---|---|
| 5 | 458.77 | 6.3e-97 |
| 10 | 1061.87 | 8.7e-222 |
| 20 | 1296.38 | 1.8e-262 |

**Engle's ARCH-LM test (lags=10):** LM = 491.55, p = 2.8e-99.

The argmax of the squared residuals lands on **2020-03-23** — the worst day of the COVID crash, which is exactly the kind of cluster-anchoring observation GARCH is designed for. Sprint 04's hand-wave that "ARIMA leaves variance structure on the table" is now a quantitative fact: the variance is decisively autocorrelated.

## GARCH(1,1) parameters

| Parameter | Value | Interpretation |
|---|---|---|
| μ | +0.000877 | Daily drift on log-return scale (matches Sprint 04's ARIMA drift of +0.0009 ✓) |
| ω | 1.38e-05 | Long-run variance intercept |
| α | 0.0793 | ARCH term — how much yesterday's surprise matters for today's variance |
| β | 0.8734 | GARCH term — how much yesterday's variance carries into today |
| α + β | **0.9527** | Persistence — volatility-shock half-life ≈ ln(0.5)/ln(0.9527) ≈ 14 trading days |

α + β = 0.9527 is the headline number. *α + β close to 1 means volatility shocks are very persistent — common in equity markets. This is why simple ARIMA on returns misses a lot of the story.* It's not into IGARCH territory (>0.97) but it's solidly in the "volatility is sticky" band typical of daily equity returns. A spike like the COVID bar takes about two weeks of half-lives to decay back to normal — which is what you see in the figure (the navy σ_t curve takes most of 2020-Q3 to walk down from the March/April spike).

The mean μ = +0.000877 from this fit lines up nicely with Sprint 04's ARIMA-implied drift of +0.0009. Good cross-check: two different models, same effective constant.

α and β are individually well-identified — β has t = 18.6 (p ≈ 3e-77), α has t = 2.79 (p ≈ 0.005). ω is borderline (p = 0.087); on this dataset ω is small relative to the cluster magnitudes, so the optimiser has weaker signal for it. Doesn't matter much for forecasts because the recursion is dominated by the α and β terms once you're past the first few steps.

## Test-period forecast

| | mean | min | max |
|---|---|---|---|
| Predicted σ_t | 0.01479 | 0.01151 | 0.02909 |
| Realised |return| | 0.00992 | — | 0.07780 |

Predicted σ slightly above realised |return| — expected behaviour. σ is the standard deviation of the *distribution* of returns, while |return| is one realised draw, so on average |return| ≈ 0.8 × σ for a normal (E[|N(0,σ²)|] = σ·√(2/π) ≈ 0.798·σ). 0.00992 / 0.01479 ≈ 0.67, slightly below the normal value — consistent with the well-known fat-tail story (most days have |return| smaller than σ predicts, but the rare outlier days are bigger than σ predicts; the mean averages low). The realised max of 0.0778 vs predicted max of 0.0291 makes the same point: a normal-innovation GARCH systematically underestimates extreme tails. A fat-tail innovation distribution (Student-t, GED) would tighten this — flagged below as an obvious follow-up.

## Sanity checks

- ✓ μ from GARCH (+0.000877) matches ARIMA drift (+0.0009 from Sprint 04). Two independent fits agree.
- ✓ α + β < 1 ⇒ stationary GARCH (the unconditional variance exists).
- ✓ Squared-residual argmax = 2020-03-23 — COVID crash day, sanity-checks the data alignment.
- ✓ In-sample σ_t length (1907) = in-sample return length. Test σ_t length (562) = test return length.
- ✓ Test forecast saved at `data/processed/garch_volatility_forecast.csv`, 562 rows, no NaNs.
- ✓ Figure renders correctly, navy in-sample / darkred test / red shaded turbulent regime visible at default zoom.

## Dead ends / things tried

- **Considered Student-t innovations.** GARCH(1,1) with normal innovations clearly underestimates the test-period max |return| (0.0778 vs predicted σ max 0.0291). A Student-t distribution would absorb the excess kurtosis (Sprint 01 measured ≈ 9.2) and give heavier-tailed forecasts. Skipped here because the directive asks specifically for the textbook GARCH(1,1) starting point — and adding the Student-t would muddle the comparison-with-ARIMA narrative. Worth revisiting if Sprint 08's benchmark cares about tail-quantile metrics rather than mean σ forecast accuracy.
- **Returns × 100 rescaling.** The `arch` library's optimiser misbehaves on log returns at native scale (~10⁻²) — the typical recommendation in `arch`'s docs is to feed in percent returns for numerical stability. Did this and back-transformed parameters at the end (μ rescales by 1/100, ω by 1/10000, α and β are unitless). Documented in a comment so the next reader doesn't get confused by "why is μ = 0.0877 in the summary table but 0.000877 in the saved params CSV".
- **Considered exponentially-weighted in-sample fit instead of MLE.** Skipped — MLE is the textbook starting point and the directive points to it. EWMA is a Sprint 8 footnote at most.

## Queued for the next sprint

`scripts/06_mlp.py` — the first neural network. Architecture per directive: 20-day log-return window flat input → 64 ReLU → 32 ReLU → 16 ReLU → 1 linear, trained in PyTorch with Adam + MSE, 100 epochs max, batch 32, early stopping on val loss with patience 10. Uses the scaled npz arrays + `make_windows` from Sprint 03. Outputs: training-curve figure, forecast figure on RELIANCE, metrics CSV across all 7 tickers (same `full_test` / `regime_calm` / `regime_turbulent` slicing as ARIMA), and saved model checkpoint at `results/mlp_reliance.pt`. Maps to Bishop PRML Ch 5 — gradient descent + backprop are the things to demonstrate clearly.
