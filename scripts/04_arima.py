"""
04_arima.py — ARIMA / SARIMA modelling, the classical-stats backbone for the
RTSM half of this project. Maps to Brockwell & Davis Ch 3, 5, 6 and Shumway &
Stoffer Ch 3 (Box–Jenkins methodology end-to-end).

What this script does:

1. RELIANCE deep-dive: manual ARIMA order ID off the Sprint 01 ACF/PACF, an
   auto_arima cross-check, residual diagnostics (residual plot, ACF, histogram,
   Q-Q, Ljung-Box), a walk-forward one-step-ahead forecast on the test window
   back-transformed to price level, and a SARIMA-vs-ARIMA AIC comparison at
   m=5 (weekly trading cycle).
2. Summary mode for the other six tickers — auto_arima + walk-forward + metrics
   only, no diagnostic figures.
3. Per-ticker metrics (RMSE, MAE, MAPE, Directional Accuracy) sliced by
   `full_test`, `regime_calm`, `regime_turbulent` (the last column will be NaN
   on this dataset because Sprint 02's k=2 labelling puts zero turbulent days
   in the test window — that's a real finding, not a bug, and it's flagged in
   the sprint report).

Outputs:
    results/figures/arima_reliance_diagnostics.png
    results/figures/arima_reliance_forecast.png
    results/metrics/arima_metrics.csv
    data/processed/arima_reliance_residuals.csv     (consumed by 05_garch.py)

Runnable standalone (`python scripts/04_arima.py`). The helpers below
(`walk_forward_forecast`, `compute_metrics`, `directional_accuracy`,
`back_transform_to_price`) are reusable from Sprints 05/08, but importing
this module by name is awkward because it starts with a digit — the standard
trick is `importlib.util.spec_from_file_location(...)` (same as how 03's
`make_windows` ends up consumed by 06/07).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

# seed for reproducibility — auto_arima's stepwise search is deterministic but
# any future stochastic step (e.g. random_state in pmdarima 2.x) should pin to 42
np.random.seed(42)

# Suppress the convergence/FutureWarning noise statsmodels and pmdarima emit
# on every fit. We're not silencing real errors — uncomment these to debug.
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
REGIMES = ROOT / "data" / "processed" / "regime_labels.csv"
ALIGNED = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
for d in (PROC_DIR, FIG_DIR, MET_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]
HERO = "RELIANCE"
OTHERS = [t for t in TICKERS if t != HERO]

# Same boundaries as Sprint 03. ARIMA combines train+val into one in-sample
# fit window — there's no early-stopping use for a held-out val set the way
# the neural nets need it. The test window stays identical so the cross-model
# comparison in Sprint 08 stays apples-to-apples.
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Loaders — small helpers, same idiom as 03_preprocessing.py
# ---------------------------------------------------------------------------
def load_log_returns() -> pd.DataFrame:
    df = pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"]).sort_index()
    return df[TICKERS]


def load_regimes() -> pd.DataFrame:
    return pd.read_csv(REGIMES, index_col="Date", parse_dates=["Date"]).sort_index()


def load_aligned_prices() -> pd.DataFrame:
    df = pd.read_csv(ALIGNED, index_col="Date", parse_dates=["Date"]).sort_index()
    return df[TICKERS]


def split_train_test(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """In-sample = train+val (≤ 2023-12-31), test = 2024-01-01 onwards."""
    in_sample = returns.loc[returns.index <= VAL_END]
    test = returns.loc[returns.index > VAL_END]
    return in_sample, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(sign(y_pred) == sign(y_true)).mean() — directive's exact spec."""
    return float((np.sign(y_pred) == np.sign(y_true)).mean())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """RMSE / MAE / MAPE / DA on log-return space.

    MAPE on log returns is dicey because returns can be near zero — divide-by-tiny
    will make the value noisy. Reported for completeness; treat DA and RMSE as
    the headline numbers.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "da": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # sklearn's MAPE handles the zero-division by clipping epsilon, but on
    # log returns ~0 that still produces wild values. Logged honestly, not hidden.
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    da = directional_accuracy(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "mape": mape, "da": da}


# ---------------------------------------------------------------------------
# Walk-forward one-step-ahead forecast — the workhorse
# ---------------------------------------------------------------------------
def walk_forward_forecast(
    order: tuple[int, int, int],
    in_sample: np.ndarray,
    test_series: np.ndarray,
    seasonal_order: tuple[int, int, int, int] | None = None,
    progress_label: str | None = None,
) -> np.ndarray:
    """One-step-ahead walk-forward over `test_series`.

    True walk-forward with full re-estimation at every step would mean fitting
    ~562 ARIMAs per ticker. That's wasteful — the parameters barely move when
    you add one more observation. Standard practice (e.g. Hyndman's `forecast`
    package, or any production rolling-forecast system) is to fix parameters
    at the in-sample MLE and filter through new observations as they arrive.
    statsmodels' `results.append(new_obs, refit=False)` does exactly that.

    Returns a 1-D array of one-step-ahead predictions, length == len(test_series).
    """
    in_sample = np.asarray(in_sample, dtype=float).ravel()
    test_series = np.asarray(test_series, dtype=float).ravel()

    if seasonal_order is None:
        model = ARIMA(in_sample, order=order)
    else:
        model = ARIMA(in_sample, order=order, seasonal_order=seasonal_order)
    res = model.fit()

    preds = np.empty(len(test_series))
    for i, actual in enumerate(test_series):
        # `.forecast` returns ndarray when fed an ndarray and a pandas Series
        # when fed a Series — handle both for robustness
        fc = res.forecast(steps=1)
        preds[i] = float(np.asarray(fc).ravel()[0])
        # extend with the realised value, keep coefficients fixed
        res = res.append([actual], refit=False)
        if progress_label and (i + 1) % 100 == 0:
            print(f"    [{progress_label}] step {i + 1}/{len(test_series)}")
    return preds


# ---------------------------------------------------------------------------
# Back-transform — predicted log returns → predicted price level
# ---------------------------------------------------------------------------
def back_transform_to_price(
    last_actual_price: float,
    actual_prices_prev: np.ndarray,
    pred_log_returns: np.ndarray,
) -> np.ndarray:
    """Convert one-step-ahead log-return predictions to price predictions.

    pred_price[t] = actual_price[t-1] * exp(pred_log_return[t])

    We ground each prediction in the *previous day's actual price* (not the
    previous predicted price) because we're doing one-step-ahead. Chaining
    predicted prices would compound errors and isn't what the directive's
    "back-transform to price level for plotting" asks for.

    `actual_prices_prev[i]` is the actual price on the day before test row i,
    so for test row 0 it's the last in-sample price.
    """
    return actual_prices_prev * np.exp(pred_log_returns)


# ---------------------------------------------------------------------------
# Diagnostic figure — RELIANCE only (directive step 4)
# ---------------------------------------------------------------------------
def plot_reliance_diagnostics(residuals: np.ndarray, order: tuple, save_to: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # top-left: residual time series
    ax = axes[0, 0]
    ax.plot(residuals, linewidth=0.6, color="navy")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Residuals over time")
    ax.set_xlabel("In-sample step")
    ax.set_ylabel("Residual")

    # top-right: ACF of residuals — should be flat if the model captured the
    # linear structure (Brockwell & Davis Ch 3 — residuals of a correctly
    # specified ARIMA should look like white noise)
    plot_acf(residuals, lags=40, ax=axes[0, 1], title="ACF of residuals (40 lags)")

    # bottom-left: histogram of residuals with a normal density overlay
    ax = axes[1, 0]
    ax.hist(residuals, bins=60, density=True, color="steelblue", alpha=0.7, edgecolor="white")
    xs = np.linspace(residuals.min(), residuals.max(), 400)
    ax.plot(
        xs,
        stats.norm.pdf(xs, residuals.mean(), residuals.std()),
        color="darkred",
        linewidth=1.5,
        label=f"Normal({residuals.mean():.1e}, {residuals.std():.1e})",
    )
    ax.set_title("Residual distribution vs normal")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right", fontsize=9)

    # bottom-right: Q-Q plot — fat tails (which we already saw in EDA's
    # excess-kurtosis ≈ 9.2) will show up as deviations from the red line in
    # the tails. Expected for daily equity returns.
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Normal Q-Q plot of residuals")
    axes[1, 1].get_lines()[1].set_color("darkred")

    fig.suptitle(f"RELIANCE — ARIMA{order} residual diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved diagnostics → {save_to}")


def plot_reliance_forecast(
    test_dates: pd.DatetimeIndex,
    actual_price: np.ndarray,
    pred_price: np.ndarray,
    order: tuple,
    save_to: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(test_dates, actual_price, color="navy", linewidth=1.2, label="Actual RELIANCE Adj Close")
    ax.plot(
        test_dates,
        pred_price,
        color="darkred",
        linewidth=1.0,
        linestyle="--",
        label=f"ARIMA{order} one-step-ahead",
    )
    ax.set_title(
        f"RELIANCE — ARIMA{order} walk-forward forecast on test window\n"
        f"(predicted price = previous actual close × exp(predicted log return))"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close (₹)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved forecast plot → {save_to}")


# ---------------------------------------------------------------------------
# Per-ticker metrics-by-regime helper — used in the final CSV assembly
# ---------------------------------------------------------------------------
def metrics_by_regime(
    test_dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: pd.DataFrame,
) -> dict[str, dict]:
    """Return three metric dicts keyed by full_test / regime_calm / regime_turbulent."""
    full = compute_metrics(y_true, y_pred)

    # Align regimes to test dates. Inner join means dates without a regime
    # label drop out — by construction the test window is fully labelled
    # (the warm-up gap is in 2016).
    aligned = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred}, index=test_dates
    ).join(regime_labels[["regime"]], how="inner")
    calm_mask = aligned["regime"] == 0
    turb_mask = aligned["regime"] == 1
    calm = compute_metrics(aligned.loc[calm_mask, "y_true"].values, aligned.loc[calm_mask, "y_pred"].values)
    turb = compute_metrics(aligned.loc[turb_mask, "y_true"].values, aligned.loc[turb_mask, "y_pred"].values)
    # On this dataset turb_mask sums to 0 → turb is a dict of NaNs. That's the
    # correct answer per the Sprint 02 finding (test window is 100% calm).
    return {"full_test": full, "regime_calm": calm, "regime_turbulent": turb}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading log returns from {LOG_RET}")
    returns = load_log_returns()
    print(
        f"  shape={returns.shape}, "
        f"range={returns.index.min().date()} → {returns.index.max().date()}"
    )

    print(f"Loading regime labels from {REGIMES}")
    regimes = load_regimes()
    print(f"  {len(regimes)} labelled rows")

    print(f"Loading aligned prices from {ALIGNED} (for back-transform)")
    prices = load_aligned_prices()

    in_sample, test = split_train_test(returns)
    print(
        f"In-sample (train+val): {len(in_sample):4d} rows  "
        f"{in_sample.index.min().date()} → {in_sample.index.max().date()}"
    )
    print(
        f"Test:                  {len(test):4d} rows  "
        f"{test.index.min().date()} → {test.index.max().date()}"
    )

    # =========================================================================
    # RELIANCE deep-dive (directive steps 1–6)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RELIANCE deep-dive")
    print("=" * 70)

    rel_in = in_sample[HERO].values
    rel_test = test[HERO].values
    test_dates = test.index

    # ---- Step 1: manual order identification --------------------------------
    # Looking at results/figures/reliance_acf_pacf.png from Sprint 01: both ACF
    # and PACF on RELIANCE log returns sit inside the confidence band at almost
    # every lag — basically white noise with maybe a faint hint at lag 7 and 11.
    # That's textbook for daily equity returns (Brockwell & Davis Ch 3 explicitly
    # warn that they're often very close to white noise). A small ARIMA(1, 0, 1)
    # is a defensible manual guess: one AR and one MA term to capture any tiny
    # short-lag structure, d=0 because EDA's ADF/KPSS already confirmed log
    # returns are stationary. I expect auto_arima to land on something equally
    # parsimonious or even (0, 0, 0).
    manual_order = (1, 0, 1)
    print(f"\n[1] Manual order from ACF/PACF (Sprint 01 figure): ARIMA{manual_order}")
    print("    ACF & PACF are essentially flat past lag 0 — daily equity returns are")
    print("    close to white noise. A small (1,0,1) is a safe manual proposal.")

    # ---- Step 2: auto_arima cross-check -------------------------------------
    print("\n[2] Running auto_arima (non-seasonal) for RELIANCE...")
    auto_model = auto_arima(
        rel_in,
        seasonal=False,
        stepwise=True,
        trace=False,
        suppress_warnings=True,
        error_action="ignore",
        max_p=5,
        max_q=5,
        d=0,
        information_criterion="aic",
    )
    auto_order = auto_model.order
    auto_aic = auto_model.aic()
    print(f"    auto_arima selected ARIMA{auto_order}, AIC={auto_aic:.2f}")
    if auto_order == manual_order:
        print("    Same as manual guess — convenient.")
    else:
        # honest comment per the directive's exact phrasing
        print(
            f"    auto_arima suggested {auto_order} — slightly different from my manual"
            f" read, probably because it's optimising AIC more rigorously."
        )

    final_order = auto_order  # trust auto_arima for the final fit

    # ---- Step 3: fit final ARIMA on in-sample (train+val) -------------------
    print(f"\n[3] Fitting final ARIMA{final_order} on in-sample (train+val)...")
    final_fit = ARIMA(rel_in, order=final_order).fit()
    print(final_fit.summary())

    # ---- Step 4: residual diagnostics ---------------------------------------
    print("\n[4] Residual diagnostics")
    residuals = np.asarray(final_fit.resid, dtype=float)

    # Save residuals for Sprint 05 (GARCH consumes these). Indexed by in-sample
    # date so the GARCH script can join them back with returns/regimes if needed.
    res_series = pd.Series(residuals, index=in_sample.index, name="residual")
    res_path = PROC_DIR / "arima_reliance_residuals.csv"
    res_series.to_frame().to_csv(res_path)
    print(f"    Saved residuals ({len(res_series)} rows) → {res_path}")

    # Ljung-Box at lags 10 and 20. Null: residuals have no autocorrelation up
    # to that lag. Want p > 0.05 → cannot reject white-noise null → ARIMA
    # captured the linear structure.
    lb = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    print("    Ljung-Box test on residuals:")
    for lag, row in lb.iterrows():
        verdict = "white-noise-like ✓" if row["lb_pvalue"] > 0.05 else "still autocorrelated ✗"
        print(f"      lag={lag:2d}: stat={row['lb_stat']:.3f}, p={row['lb_pvalue']:.4f}  ({verdict})")
    # If lag-20 p < 0.05 we'd note that ARIMA hasn't captured everything — but
    # that's not surprising for daily equity returns; what's left is mostly
    # variance structure (heteroskedasticity) which is exactly what GARCH
    # (Sprint 05) picks up.

    plot_reliance_diagnostics(
        residuals, final_order, FIG_DIR / "arima_reliance_diagnostics.png"
    )

    # ---- Step 5: walk-forward one-step-ahead forecast on test ---------------
    print("\n[5] Walk-forward one-step-ahead forecast on test window...")
    rel_pred = walk_forward_forecast(
        final_order, rel_in, rel_test, progress_label="RELIANCE"
    )

    # Back-transform to price level. actual_prices_prev[i] = price on the day
    # *before* test row i. For test row 0 that's the last in-sample price.
    # The aligned_prices frame has the actual close for every trading day, so
    # we just shift by one.
    rel_actual_price_test = prices.loc[test_dates, HERO].values
    last_in_sample_price = float(prices.loc[in_sample.index[-1], HERO])
    actual_prev = np.concatenate([[last_in_sample_price], rel_actual_price_test[:-1]])
    rel_pred_price = back_transform_to_price(last_in_sample_price, actual_prev, rel_pred)

    # Honest note: a one-step-ahead price forecast plotted against the actual
    # price always tracks closely — most of the "prediction" is yesterday's
    # actual close, and the predicted log return is small. Don't over-read
    # this figure; the metrics CSV is the real verdict.
    plot_reliance_forecast(
        test_dates,
        rel_actual_price_test,
        rel_pred_price,
        final_order,
        FIG_DIR / "arima_reliance_forecast.png",
    )

    # ---- Step 6: SARIMA seasonality check at m=5 (weekly) ------------------
    print("\n[6] SARIMA check at m=5 (weekly trading cycle)...")
    try:
        sarima_model = auto_arima(
            rel_in,
            seasonal=True,
            m=5,
            stepwise=True,
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3,
            max_q=3,
            max_P=2,
            max_Q=2,
            d=0,
            D=0,
            information_criterion="aic",
        )
        sarima_aic = sarima_model.aic()
        print(
            f"    SARIMA: order={sarima_model.order}, "
            f"seasonal_order={sarima_model.seasonal_order}, AIC={sarima_aic:.2f}"
        )
        delta = auto_aic - sarima_aic
        if delta < 5:
            # Directive's exact phrasing for the (expected) negative result:
            print(
                "    tried adding a seasonal component with m=5 (weekly), but it didn't"
                " improve AIC meaningfully — makes sense, daily stock returns don't have"
                " strong weekly seasonality."
            )
            print(f"    AIC delta (non-seasonal − seasonal) = {delta:+.2f}")
        else:
            print(f"    SARIMA materially improves AIC by {delta:.2f} — worth keeping.")
            print("    !! NOTE: SARIMA path now beats non-seasonal — revisit downstream scripts.")
    except Exception as e:
        print(f"    SARIMA fit failed: {e}. Sticking with non-seasonal ARIMA.")

    # =========================================================================
    # Summary mode for the other six tickers (directive step 7)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary-mode auto_arima + walk-forward for remaining tickers")
    print("=" * 70)

    per_ticker_results: dict[str, dict] = {}
    per_ticker_results[HERO] = {
        "order": final_order,
        "y_true": rel_test,
        "y_pred": rel_pred,
    }

    for ticker in OTHERS:
        print(f"\n  {ticker}: fitting auto_arima...")
        in_arr = in_sample[ticker].values
        test_arr = test[ticker].values

        am = auto_arima(
            in_arr,
            seasonal=False,
            stepwise=True,
            trace=False,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5,
            max_q=5,
            d=0,
            information_criterion="aic",
        )
        order = am.order
        print(f"    order={order}, AIC={am.aic():.2f}")

        print(f"  {ticker}: walk-forward forecast on {len(test_arr)} test steps...")
        pred = walk_forward_forecast(order, in_arr, test_arr, progress_label=ticker)
        m = compute_metrics(test_arr, pred)
        print(
            f"    {ticker}: order={order}  "
            f"RMSE={m['rmse']:.5f}  MAE={m['mae']:.5f}  DA={m['da']:.3f}"
        )

        per_ticker_results[ticker] = {
            "order": order,
            "y_true": test_arr,
            "y_pred": pred,
        }

    # =========================================================================
    # Metrics CSV (directive step 8)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Assembling metrics CSV (sliced by full_test / regime_calm / regime_turbulent)")
    print("=" * 70)

    rows: list[dict] = []
    for ticker in TICKERS:
        r = per_ticker_results[ticker]
        sliced = metrics_by_regime(test_dates, r["y_true"], r["y_pred"], regimes)
        for metric in ("rmse", "mae", "mape", "da"):
            rows.append(
                {
                    "ticker": ticker,
                    "order": str(r["order"]),
                    "metric": metric,
                    "full_test": sliced["full_test"][metric],
                    "regime_calm": sliced["regime_calm"][metric],
                    "regime_turbulent": sliced["regime_turbulent"][metric],
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_path = MET_DIR / "arima_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics → {metrics_path}  (shape={metrics_df.shape})")
    print(metrics_df.to_string(index=False))

    # Headline summary line for the run log
    da_table = metrics_df[metrics_df["metric"] == "da"][["ticker", "full_test"]]
    print("\nDirectional accuracy by ticker (full test):")
    for _, row in da_table.iterrows():
        print(f"  {row['ticker']:8s} {row['full_test']:.3f}")

    print("\nDone. Outputs in results/figures/, results/metrics/, data/processed/.")


if __name__ == "__main__":
    main()
