"""
05_garch.py — GARCH(1,1) on RELIANCE log returns. Models the *variance*
structure that ARIMA leaves on the table.

ARIMA(0,0,0) in Sprint 04 essentially said "tomorrow's return ≈ historical
mean drift + white noise" — and the Ljung-Box on residuals failed at lag 10
and lag 20 (p ≈ 0.007 / 0.0004), confirming there's leftover autocorrelation
the mean model didn't catch. A lot of that "leftover" is actually variance
clustering — calm days follow calm days, turbulent days follow turbulent
days — which is exactly what GARCH is built for. Maps to Brockwell & Davis
Ch 4 on heteroskedasticity.

What this script does:

1. Load the in-sample residuals saved by Sprint 04 plus the in-sample log
   returns. Plot squared residuals — volatility clustering should be visually
   obvious (and it is, especially around the COVID stretch ~step 1000).
2. Run two ARCH-effect tests on the residuals:
   - Ljung-Box on squared residuals (autocorrelation in the variance)
   - Engle's ARCH-LM test via `statsmodels.stats.diagnostic.het_arch`
3. Fit GARCH(1,1) via the `arch` library on the in-sample log returns. Use a
   constant mean (the Sprint 04 ARIMA found a small but statistically
   significant +0.0009/day drift, so 'Constant' is more honest than 'Zero')
   and a normal innovation distribution as the textbook starting point.
4. Walk-forward one-step-ahead variance forecast on the test window using the
   same fixed-parameters trick as Sprint 04: parameters pinned at in-sample
   MLE, the GARCH recursion σ²_{t+1} = ω + α·ε²_t + β·σ²_t is applied with
   realised test-window returns. Equivalent to refit=False.
5. Save α, β, ω, persistence (= α+β) to results/metrics/garch_params.csv. Save
   the test-period volatility forecast to data/processed/ for Sprint 08.
6. Plot conditional volatility over the full period (in-sample + test forecast)
   with the COVID turbulent regime shaded and |log returns| overlaid.

Outputs:
    results/figures/garch_volatility.png
    results/metrics/garch_params.csv
    data/processed/garch_volatility_forecast.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# seed for reproducibility — arch's optimiser is deterministic but every
# script in this repo sets it at the top by convention
np.random.seed(42)

# Quiet the convergence/ConstantMean warnings — not silencing real errors.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
REGIMES = ROOT / "data" / "processed" / "regime_labels.csv"
RESIDS = ROOT / "data" / "processed" / "arima_reliance_residuals.csv"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
for d in (PROC_DIR, FIG_DIR, MET_DIR):
    d.mkdir(parents=True, exist_ok=True)

HERO = "RELIANCE"

# Same boundaries as Sprint 04 — train+val combined as in-sample, test
# starts 2024-01-01. GARCH has no held-out-val use either; the train/val
# distinction is for the neural nets.
VAL_END = pd.Timestamp("2023-12-31")

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_returns_for_hero() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (full_returns, in_sample, test) for the hero ticker."""
    df = pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"]).sort_index()
    full = df[HERO].dropna()
    in_sample = full.loc[full.index <= VAL_END]
    test = full.loc[full.index > VAL_END]
    return full, in_sample, test


def load_residuals() -> pd.Series:
    s = pd.read_csv(RESIDS, index_col=0, parse_dates=[0])
    return s["residual"].astype(float)


def load_regimes() -> pd.DataFrame:
    return pd.read_csv(REGIMES, index_col="Date", parse_dates=["Date"]).sort_index()


# ---------------------------------------------------------------------------
# ARCH-effect tests
# ---------------------------------------------------------------------------
def test_arch_effects(residuals: np.ndarray) -> None:
    """Two complementary tests for variance autocorrelation."""
    # the Ljung-Box test on squared residuals tests for ARCH effects — basically
    # asking if there's autocorrelation in the variance, not just the mean.
    # Null: squared residuals are uncorrelated (no ARCH effect). Reject (small
    # p-value) → variance clustering is present, GARCH is justified.
    lb = acorr_ljungbox(residuals**2, lags=[5, 10, 20], return_df=True)
    print("\nLjung-Box on squared residuals (null = no ARCH effect):")
    for lag, row in lb.iterrows():
        verdict = "ARCH effect present ✓" if row["lb_pvalue"] < 0.05 else "no ARCH detected"
        print(
            f"  lag={lag:2d}: stat={row['lb_stat']:.3f}, p={row['lb_pvalue']:.3g}  ({verdict})"
        )

    # Engle's ARCH-LM test — same null but framed as a regression of squared
    # residuals on their own lags. Engle 1982. statsmodels gives back
    # (LM_stat, LM_pvalue, F_stat, F_pvalue).
    lm_stat, lm_p, f_stat, f_p = het_arch(residuals, nlags=10)
    print("\nEngle's ARCH-LM test (lags=10):")
    print(f"  LM stat  = {lm_stat:.3f}, p = {lm_p:.3g}")
    print(f"  F stat   = {f_stat:.3f}, p = {f_p:.3g}")
    if lm_p < 0.05:
        print("  → reject 'no ARCH' null. Variance is autocorrelated. GARCH is justified.")
    else:
        print("  → cannot reject 'no ARCH' null. GARCH may be unnecessary on this data.")


# ---------------------------------------------------------------------------
# Walk-forward one-step-ahead variance forecast
# ---------------------------------------------------------------------------
def walk_forward_variance(
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    last_in_sample_eps2: float,
    last_in_sample_sigma2: float,
    test_returns: np.ndarray,
) -> np.ndarray:
    """One-step-ahead conditional variance via the GARCH recursion.

    σ²_{t+1} = ω + α·ε²_t + β·σ²_t

    Same trick as Sprint 04's ARIMA walk-forward: parameters fixed at in-sample
    MLE, realised returns fed in as observations to update ε_t. Equivalent to
    `refit=False` rolling forecast. Returns a 1-D array of conditional
    variance forecasts σ²_{t} for each test step.
    """
    test_returns = np.asarray(test_returns, dtype=float).ravel()
    sigma2_preds = np.empty(len(test_returns))
    eps2_prev = last_in_sample_eps2
    sigma2_prev = last_in_sample_sigma2
    for i, r in enumerate(test_returns):
        sigma2_next = omega + alpha * eps2_prev + beta * sigma2_prev
        sigma2_preds[i] = sigma2_next
        # observe the realised return → update ε for next step
        eps_now = r - mu
        eps2_prev = eps_now**2
        sigma2_prev = sigma2_next
    return sigma2_preds


# ---------------------------------------------------------------------------
# Plot — conditional vol over full period + test forecast + regime shading
# ---------------------------------------------------------------------------
def plot_volatility(
    full_returns: pd.Series,
    in_sample_sigma: pd.Series,
    test_sigma_forecast: pd.Series,
    regimes: pd.DataFrame,
    save_to: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    # |log returns| as the noisy backdrop — the thing GARCH is trying to track
    abs_ret = full_returns.abs()
    ax.plot(
        abs_ret.index,
        abs_ret.values,
        color="lightgrey",
        linewidth=0.6,
        label="|log return|",
    )

    # In-sample conditional volatility (one σ, not σ²)
    ax.plot(
        in_sample_sigma.index,
        in_sample_sigma.values,
        color="navy",
        linewidth=1.2,
        label="In-sample σ_t (GARCH(1,1))",
    )

    # Test-period one-step-ahead volatility forecast
    ax.plot(
        test_sigma_forecast.index,
        test_sigma_forecast.values,
        color="darkred",
        linewidth=1.2,
        linestyle="--",
        label="Test-period σ_t (one-step-ahead, params fixed)",
    )

    # Train/test boundary
    boundary = test_sigma_forecast.index.min()
    ax.axvline(boundary, color="black", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(
        boundary,
        ax.get_ylim()[1] * 0.95,
        "  train/test",
        fontsize=9,
        color="black",
        verticalalignment="top",
    )

    # Shade the turbulent regime (the COVID window — only one contiguous span
    # under k=2; see Sprint 02 report)
    turb_dates = regimes.loc[regimes["regime"] == 1].index
    if len(turb_dates) > 0:
        # span from first to last turbulent label is contiguous on this dataset
        ax.axvspan(
            turb_dates.min(),
            turb_dates.max(),
            color="red",
            alpha=0.10,
            label="Turbulent regime (COVID, Sprint 02)",
        )

    ax.set_title(
        "RELIANCE — GARCH(1,1) conditional volatility σ_t\n"
        "(in-sample fit + walk-forward test forecast)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility / |log return|")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    full_ret, in_sample_ret, test_ret = load_returns_for_hero()
    print(f"RELIANCE log returns: full={len(full_ret)}, in_sample={len(in_sample_ret)}, test={len(test_ret)}")
    print(
        f"  in-sample: {in_sample_ret.index.min().date()} → {in_sample_ret.index.max().date()}"
    )
    print(
        f"  test:      {test_ret.index.min().date()} → {test_ret.index.max().date()}"
    )

    residuals = load_residuals()
    print(f"Loaded ARIMA residuals: {len(residuals)} rows from {RESIDS}")
    assert len(residuals) == len(in_sample_ret), \
        "residuals length should match in-sample length — Sprint 04 saved them on the in-sample window"

    regimes = load_regimes()
    print(f"Loaded regime labels: {len(regimes)} rows from {REGIMES}")

    # -------------------------------------------------------------------
    # Step 1+2: visual + statistical ARCH-effect checks
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ARCH-effect checks on Sprint 04 residuals")
    print("=" * 70)

    print(f"  Squared residuals: mean={np.mean(residuals**2):.6f}, "
          f"max={np.max(residuals**2):.6f}, "
          f"argmax_date={residuals.index[np.argmax(residuals**2)].date()}")
    # If the argmax date sits in March 2020, that's the COVID single-day move
    # — exactly the kind of cluster GARCH wants to model.
    test_arch_effects(residuals.values)

    # -------------------------------------------------------------------
    # Step 3: fit GARCH(1,1)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Fitting GARCH(1,1) on RELIANCE in-sample log returns")
    print("=" * 70)
    # arch's `arch_model` rescales returns internally if asked. We pass the
    # raw log returns × 100 — the library's own docs recommend percent returns
    # for numerical stability of the optimiser. We rescale back at the end so
    # the saved σ values are on the same scale as the log returns themselves.
    SCALE = 100.0
    am = arch_model(
        in_sample_ret.values * SCALE,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
    )
    res = am.fit(disp="off", show_warning=False)
    print(res.summary())

    # Pull params out. arch's parameter names: 'mu' (Constant mean), 'omega',
    # 'alpha[1]', 'beta[1]'. They're estimated on returns × SCALE, so we
    # rescale back below. (μ rescales by 1/SCALE, ω by 1/SCALE², α and β are
    # unitless.)
    mu_scaled = float(res.params["mu"])
    omega_scaled = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    mu = mu_scaled / SCALE
    omega = omega_scaled / (SCALE**2)
    persistence = alpha + beta

    print("\nParameters (back-transformed to original log-return scale):")
    print(f"  μ          = {mu:+.6f}")
    print(f"  ω          = {omega:.3e}")
    print(f"  α          = {alpha:.4f}")
    print(f"  β          = {beta:.4f}")
    print(f"  α + β      = {persistence:.4f}  (persistence)")
    # alpha + beta close to 1 means volatility shocks are very persistent —
    # common in equity markets. This is why simple ARIMA on returns misses a
    # lot of the story.
    if persistence > 0.97:
        print(
            "  → α+β > 0.97. Volatility is extremely sticky here, near-IGARCH"
            " territory. Equity-returns norm."
        )
    elif persistence > 0.90:
        print("  → α+β > 0.90. Persistent volatility, classic equity behaviour.")
    else:
        print("  → α+β < 0.90. Volatility shocks decay faster than typical.")

    # In-sample conditional volatility (σ_t), back to original return scale
    in_sample_sigma = pd.Series(
        res.conditional_volatility / SCALE, index=in_sample_ret.index, name="sigma"
    )

    # -------------------------------------------------------------------
    # Step 4: walk-forward variance forecast on test
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Walk-forward one-step-ahead variance forecast on test window")
    print("=" * 70)

    last_eps = (in_sample_ret.iloc[-1] - mu)
    last_sigma2 = float(in_sample_sigma.iloc[-1] ** 2)
    test_sigma2 = walk_forward_variance(
        omega=omega,
        alpha=alpha,
        beta=beta,
        mu=mu,
        last_in_sample_eps2=last_eps**2,
        last_in_sample_sigma2=last_sigma2,
        test_returns=test_ret.values,
    )
    test_sigma = pd.Series(np.sqrt(test_sigma2), index=test_ret.index, name="sigma")
    print(
        f"  Test-period σ_t: mean={test_sigma.mean():.5f}, "
        f"min={test_sigma.min():.5f}, max={test_sigma.max():.5f}"
    )
    print(
        f"  Realised |return| over test: mean={test_ret.abs().mean():.5f}, "
        f"max={test_ret.abs().max():.5f}"
    )

    # Save the test-period forecast for Sprint 08 (regime-conditioned eval)
    test_sigma.to_frame().to_csv(PROC_DIR / "garch_volatility_forecast.csv")
    print(f"  Saved test-period vol forecast → {PROC_DIR / 'garch_volatility_forecast.csv'}")

    # -------------------------------------------------------------------
    # Step 5: params CSV
    # -------------------------------------------------------------------
    params_df = pd.DataFrame(
        [
            {"param": "mu", "value": mu, "note": "constant mean of log returns"},
            {"param": "omega", "value": omega, "note": "long-run variance intercept (back-transformed from rescaled fit)"},
            {"param": "alpha", "value": alpha, "note": "ARCH coefficient (impact of last innovation²)"},
            {"param": "beta", "value": beta, "note": "GARCH coefficient (persistence of last variance)"},
            {"param": "persistence", "value": persistence, "note": "alpha + beta; >0.97 ≈ IGARCH territory"},
            {"param": "loglik", "value": float(res.loglikelihood), "note": "log-likelihood at MLE (on rescaled returns)"},
            {"param": "aic", "value": float(res.aic), "note": "AIC (on rescaled returns)"},
            {"param": "bic", "value": float(res.bic), "note": "BIC (on rescaled returns)"},
        ]
    )
    params_path = MET_DIR / "garch_params.csv"
    params_df.to_csv(params_path, index=False)
    print(f"\nSaved params → {params_path}")
    print(params_df.to_string(index=False))

    # -------------------------------------------------------------------
    # Step 6: plot
    # -------------------------------------------------------------------
    print("\nPlotting conditional volatility over full period...")
    plot_volatility(
        full_returns=full_ret,
        in_sample_sigma=in_sample_sigma,
        test_sigma_forecast=test_sigma,
        regimes=regimes,
        save_to=FIG_DIR / "garch_volatility.png",
    )

    print("\nDone. Outputs in results/figures/, results/metrics/, data/processed/.")


if __name__ == "__main__":
    main()
