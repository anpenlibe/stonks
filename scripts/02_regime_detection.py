"""
02_regime_detection.py — Discover market regimes via k-means clustering.

Instead of cherry-picking calm vs turbulent windows by eye (e.g. "everything
around March 2020 is turbulent"), we let the data decide. We build a few
rolling-window features that summarise local volatility and drift, run k-means
across a grid of k, plot an elbow curve, and then commit to k=2 as the directive
spec asks for. The two clusters get relabelled by their *raw* RELIANCE rolling
volatility so 0 is always the calmer regime and 1 is always the more turbulent
one — that way every downstream script can rely on the same label semantics.

Maps to the clustering material from the AI/ML course (Bishop PRML Ch 9 covers
mixture models, but k-means is the simpler cousin we did in class). Interestingly,
using k-means to find regimes means we let the data decide the boundaries rather
than cherry-picking dates — feels more honest statistically.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# seed for reproducibility — k-means initialisation is random, this pins it
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
PROC_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]
HERO = "RELIANCE"
WINDOW = 60  # 60 trading days ≈ 3 months — same window the EDA script used

sns.set_theme(style="whitegrid")


def load_prices() -> pd.DataFrame:
    df = pd.read_csv(DATA, index_col="Date", parse_dates=["Date"]).sort_index()
    return df[TICKERS]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff().dropna(how="all")


# ---------------------------------------------------------------------------
# Feature engineering — rolling vol + drift on RELIANCE plus market-wide vol
# ---------------------------------------------------------------------------
def build_features(returns: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    # The intuition: a "regime" is a stretch of days where local volatility and
    # drift have a similar character. Rolling vol of RELIANCE captures the
    # idiosyncratic story; rolling vol of NSEI captures the broader market mood.
    feats = pd.DataFrame(
        {
            "reliance_vol": returns[HERO].rolling(window).std(),
            "reliance_mean": returns[HERO].rolling(window).mean(),
            "nsei_vol": returns["NSEI"].rolling(window).std(),
        }
    ).dropna()
    print(f"Built {len(feats)} feature rows ({window}-day rolling window).")
    return feats


# ---------------------------------------------------------------------------
# Elbow curve — fit k=2..8 on standardised features, plot inertia vs k
# ---------------------------------------------------------------------------
def plot_elbow(features: pd.DataFrame) -> None:
    print("Fitting k-means for k=2..8 to draw the elbow curve...")
    # IMPORTANT: standardise before clustering — k-means is distance-based, so
    # if we left reliance_vol on its raw 0.01-ish scale and reliance_mean on its
    # 0.001-ish scale, the bigger-magnitude feature would silently dominate the
    # cluster geometry. Classic clustering gotcha covered in class.
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    ks = list(range(2, 9))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k}: inertia={km.inertia_:.1f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, inertias, marker="o", linewidth=1.6, color="navy")
    ax.set_title("K-means elbow curve on rolling-window regime features")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("inertia (within-cluster sum of squares)")
    # elbow curve to pick k — as covered in class, we look for where adding
    # another cluster stops helping much. The directive commits to k=2 anyway,
    # but the curve is the justification.
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_elbow_curve.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fit k=2, relabel clusters as calm (0) / turbulent (1) by raw rolling vol
# ---------------------------------------------------------------------------
def fit_regimes(features: pd.DataFrame) -> pd.DataFrame:
    print("Fitting k-means with k=2 for the regime labelling...")
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    raw_labels = km.fit_predict(X)

    # Decide which raw cluster is calm vs turbulent by looking at the *raw*
    # (unscaled) RELIANCE rolling vol per cluster — the cluster with the higher
    # mean vol is the turbulent one.
    df = features.copy()
    df["raw_label"] = raw_labels
    vol_by_cluster = df.groupby("raw_label")["reliance_vol"].mean().sort_values()
    calm_raw = int(vol_by_cluster.index[0])
    turbulent_raw = int(vol_by_cluster.index[1])
    print(
        f"  cluster {calm_raw} mean reliance_vol = {vol_by_cluster.iloc[0]:.4f} (→ calm)"
    )
    print(
        f"  cluster {turbulent_raw} mean reliance_vol = {vol_by_cluster.iloc[1]:.4f} (→ turbulent)"
    )

    remap = {calm_raw: 0, turbulent_raw: 1}
    labels = pd.Series(raw_labels, index=features.index).map(remap)
    out = pd.DataFrame(
        {
            "regime": labels.astype(int),
            "regime_name": labels.map({0: "calm", 1: "turbulent"}),
        }
    )
    out.index.name = "Date"
    return out


# ---------------------------------------------------------------------------
# Regime-shaded RELIANCE price plot
# ---------------------------------------------------------------------------
def plot_regime_shaded_price(prices: pd.DataFrame, labels: pd.DataFrame) -> None:
    print("Plotting RELIANCE price with regime-shaded background...")
    price = prices[HERO]

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(price.index, price.values, color="black", linewidth=0.9, label=f"{HERO} Adj Close")

    # Walk through contiguous runs of the same label and shade with axvspan.
    # We use the labels DataFrame's index (which starts after the rolling window
    # warm-up) — dates before that are simply not shaded, which honestly
    # represents that we don't have enough rolling data to assign a regime there.
    regime = labels["regime"]
    if len(regime):
        runs = (regime != regime.shift()).cumsum()
        for _, run_idx in regime.groupby(runs).groups.items():
            run_idx = pd.DatetimeIndex(run_idx)
            start, end = run_idx.min(), run_idx.max()
            label_val = regime.loc[start]
            color = "#2ca02c" if label_val == 0 else "#d62728"
            ax.axvspan(start, end, color=color, alpha=0.15)

    # Manual legend handles for the shading
    from matplotlib.patches import Patch

    handles, leg_labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="#2ca02c", alpha=0.3, label="calm regime"),
                Patch(facecolor="#d62728", alpha=0.3, label="turbulent regime")]
    ax.legend(handles=handles, loc="upper left")
    ax.set_title(f"{HERO} Adj Close with k-means regime shading (k=2)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close (INR)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "regime_detection.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary — counts, share, and longest contiguous runs per regime
# ---------------------------------------------------------------------------
def summarise(labels: pd.DataFrame, top_n_runs: int = 5) -> None:
    print("\nRegime summary")
    print("--------------")
    total = len(labels)
    counts = labels["regime"].value_counts().sort_index()
    for regime, count in counts.items():
        name = "calm" if regime == 0 else "turbulent"
        print(f"  regime {regime} ({name}): {count} days, {count / total:.1%} of total")

    regime = labels["regime"]
    runs = (regime != regime.shift()).cumsum()
    spans = []
    for _, idx in regime.groupby(runs).groups.items():
        idx = pd.DatetimeIndex(idx)
        spans.append({
            "regime": int(regime.loc[idx.min()]),
            "start": idx.min().date(),
            "end": idx.max().date(),
            "n_days": len(idx),
        })
    spans_df = pd.DataFrame(spans).sort_values("n_days", ascending=False)

    for r, name in [(0, "calm"), (1, "turbulent")]:
        sub = spans_df[spans_df["regime"] == r].head(top_n_runs)
        print(f"\n  longest {name} runs (top {top_n_runs}):")
        for _, row in sub.iterrows():
            print(f"    {row['start']} → {row['end']}  ({row['n_days']} days)")

    # The turbulent regime should naturally pick up COVID 2020 and the
    # rate-hike cycle through 2022 — sanity-check it visually.


def main() -> None:
    print(f"Loading aligned prices from {DATA}")
    prices = load_prices()
    print(f"  shape={prices.shape}, range={prices.index.min().date()} → {prices.index.max().date()}")

    returns = log_returns(prices)
    features = build_features(returns)

    plot_elbow(features)
    labels = fit_regimes(features)

    out_csv = PROC_DIR / "regime_labels.csv"
    labels.to_csv(out_csv)
    print(f"\nSaved regime labels → {out_csv}")

    plot_regime_shaded_price(prices, labels)
    summarise(labels)

    print(f"\nDone. Figures in {FIG_DIR}, labels in {PROC_DIR}.")


if __name__ == "__main__":
    main()
