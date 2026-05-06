"""
01_eda.py — Exploratory Data Analysis on the aligned price frame.

This is the first thing we do before touching any model. The goal is to look
at the data the way Brockwell & Davis Ch 1-2 and Shumway & Stoffer Ch 1
suggest: check what we have, see if the series even *can* be modelled with
ARIMA (i.e. is it stationary?), look at the distribution of returns, and
figure out the cross-ticker structure.

No models get fit here. Just plots, tests, and a couple of summary tables.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

# seed for reproducibility — results will be consistent across runs
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]
HERO = "RELIANCE"

# Placeholder regime boundaries for the price-history plot. These will get
# replaced with whatever k-means picks out in 02_regime_detection.py.
REGIME_PLACEHOLDERS = ["2017-01-01", "2020-01-01"]

sns.set_theme(style="whitegrid")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col="Date", parse_dates=["Date"]).sort_index()
    # Reorder to a friendlier display order rather than alphabetical
    return df[TICKERS]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # Standard definition: r_t = ln(P_t) - ln(P_{t-1}). Drops the first row.
    return np.log(prices).diff().dropna(how="all")


# ---------------------------------------------------------------------------
# 1. Price history (normalised to base 100 so we can put everything on one axis)
# ---------------------------------------------------------------------------
def plot_price_history(prices: pd.DataFrame) -> None:
    print("Plotting price history (normalised to base 100)...")
    base = prices.iloc[0]
    norm = prices.divide(base) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in norm.columns:
        ax.plot(norm.index, norm[col], label=col, linewidth=1.2)

    # Vertical guides for the regime boundaries — these are placeholders for now,
    # will be overwritten with k-means output once 02_regime_detection.py runs.
    for d in REGIME_PLACEHOLDERS:
        ax.axvline(pd.Timestamp(d), color="grey", linestyle="--", alpha=0.6)

    ax.set_title("Adjusted close, normalised to 100 at start (placeholder regime guides shown)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index (start = 100)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_history.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Daily log returns for the hero ticker
# ---------------------------------------------------------------------------
def plot_hero_log_returns(returns: pd.Series) -> None:
    print(f"Plotting daily log returns for {HERO}...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(returns.index, returns.values, linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(f"{HERO} Daily Log Returns — you can really see the COVID crash here")
    ax.set_xlabel("Date")
    ax.set_ylabel("log return")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliance_log_returns.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Return distribution vs a normal — first hint at fat tails
# ---------------------------------------------------------------------------
def plot_return_distribution(returns: pd.Series) -> None:
    print(f"Plotting {HERO} return distribution with a normal overlay...")
    mu, sigma = returns.mean(), returns.std()
    excess_kurt = stats.kurtosis(returns, fisher=True, bias=False)
    skew = stats.skew(returns, bias=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(returns, bins=80, kde=True, stat="density", color="steelblue", ax=ax)

    # Overlay a normal with the same mean/std — if the empirical distribution
    # has fatter tails than this, it's a sign we'll need a heavier-tailed model
    # (this is the motivation we'll come back to when we do GARCH).
    xs = np.linspace(returns.min(), returns.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="red", linestyle="--",
            label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")
    ax.set_title(
        f"{HERO} log return distribution — skew={skew:.2f}, excess kurtosis={excess_kurt:.2f}"
    )
    ax.set_xlabel("log return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliance_return_distribution.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    # Excess kurtosis well above 0 = fat tails relative to normal — this is the
    # classic stylised fact about equity returns. Worth flagging in the report.


# ---------------------------------------------------------------------------
# 4. Rolling mean + rolling std — visualises calm vs turbulent periods
# ---------------------------------------------------------------------------
def plot_rolling_stats(returns: pd.Series, window: int = 60) -> None:
    print(f"Plotting {window}-day rolling stats for {HERO}...")
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(rolling_mean, color="navy")
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_title(f"{HERO} — {window}-day rolling mean of log returns")
    axes[0].set_ylabel("rolling mean")

    axes[1].plot(rolling_std, color="darkred")
    axes[1].set_title(f"{HERO} — {window}-day rolling std of log returns (proxy for volatility)")
    axes[1].set_ylabel("rolling std")
    axes[1].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliance_rolling_stats.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. ADF + KPSS stationarity tests on every ticker, raw prices and log returns
# ---------------------------------------------------------------------------
def run_stationarity_tests(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    print("Running ADF and KPSS stationarity tests...")
    # ADF tests the null that a unit root exists — we need to reject this (low p)
    # for ARIMA to be valid. KPSS flips it: null is stationarity, so we want a
    # *high* p-value. The two together are more convincing than either alone.
    # Reference: Brockwell & Davis Ch 3, also Shumway & Stoffer Ch 1.
    rows = []

    def _kpss(series: pd.Series) -> tuple[float, float]:
        # KPSS warns about p-values being interpolated outside [0.01, 0.10] — fine,
        # we just suppress the noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p, *_ = kpss(series.dropna(), regression="c", nlags="auto")
        return stat, p

    for ticker in TICKERS:
        for label, series in [("price", prices[ticker]), ("log_return", returns[ticker])]:
            adf_stat, adf_p, *_ = adfuller(series.dropna(), autolag="AIC")
            kpss_stat, kpss_p = _kpss(series)
            rows.append(
                {
                    "ticker": ticker,
                    "series": label,
                    "adf_stat": adf_stat,
                    "adf_p": adf_p,
                    "kpss_stat": kpss_stat,
                    "kpss_p": kpss_p,
                    "adf_stationary": adf_p < 0.05,
                    "kpss_stationary": kpss_p > 0.05,
                }
            )

    table = pd.DataFrame(rows)
    out = MET_DIR / "stationarity_tests.csv"
    table.to_csv(out, index=False)
    print(f"  saved {out}")
    print(table.to_string(index=False))
    return table


# ---------------------------------------------------------------------------
# 6. ACF and PACF for the hero — input to Box-Jenkins ARIMA order ID
# ---------------------------------------------------------------------------
def plot_acf_pacf(returns: pd.Series, lags: int = 40) -> None:
    print(f"Plotting ACF/PACF for {HERO} log returns...")
    # ACF tail-off + PACF cutoff at lag p suggests AR(p); the reverse pattern
    # suggests MA(q). We'll come back to this in 04_arima.py when we manually
    # propose an order before running auto_arima.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(returns.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"{HERO} log returns — ACF (up to {lags} lags)")
    plot_pacf(returns.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{HERO} log returns — PACF (up to {lags} lags)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliance_acf_pacf.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Cross-ticker correlation of log returns
# ---------------------------------------------------------------------------
def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    print("Plotting correlation heatmap of log returns...")
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.4, ax=ax)
    ax.set_title("Pearson correlation of daily log returns")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Q-Q plot — another way to see whether returns look normal
# ---------------------------------------------------------------------------
def plot_qq(returns: pd.Series) -> None:
    print(f"Plotting Q-Q plot for {HERO} log returns...")
    fig, ax = plt.subplots(figsize=(7, 7))
    stats.probplot(returns.dropna(), dist="norm", plot=ax)
    ax.set_title(f"{HERO} log returns — Normal Q-Q plot")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliance_qq_plot.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Loading aligned prices from {DATA}")
    prices = load_prices()
    print(f"  shape={prices.shape}, range={prices.index.min().date()} → {prices.index.max().date()}")

    returns = log_returns(prices)
    hero_returns = returns[HERO].dropna()

    plot_price_history(prices)
    plot_hero_log_returns(hero_returns)
    plot_return_distribution(hero_returns)
    plot_rolling_stats(hero_returns)
    run_stationarity_tests(prices, returns)
    plot_acf_pacf(hero_returns)
    plot_correlation_heatmap(returns)
    plot_qq(hero_returns)

    print(f"\nDone. Figures in {FIG_DIR}, tables in {MET_DIR}.")


if __name__ == "__main__":
    main()
