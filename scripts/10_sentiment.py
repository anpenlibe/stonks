"""
10_sentiment.py — Sentiment extension. Fetch RELIANCE.NS headlines from
yfinance, score each one with VADER, smooth with a 5-day rolling mean,
and ask whether the smoothed sentiment correlates with next-day RELIANCE
log returns. Per directive §10 the sentiment-augmented LSTM is gated:
only train it if there's enough overlap with the existing train/val
window — otherwise skip it cleanly and report the correlation only.

Honest expectation up front: yfinance's news endpoint is short-window
(directive flagged 30–90 days; in practice, sometimes ~10 items). That
makes the correlation analysis a small-sample affair and almost always
puts the augmented-LSTM path out of reach. The whole point of the gate
is to avoid pretending we trained something meaningful on a handful of
rows.

Determinism: the raw yfinance pull is non-deterministic across calls
(new headlines arrive). We persist the first pull to
`data/processed/sentiment_raw.csv` and prefer the cache on rerun so the
downstream artefacts are byte-stable. Set the env var
`SENTIMENT_REFRESH=1` to force a re-pull.

Outputs:
    data/processed/sentiment_raw.csv         (fetched headlines + per-item scores)
    data/processed/sentiment_scores.csv      (daily + rolling compound)
    results/metrics/sentiment_correlation.csv
    results/figures/sentiment_correlation.png
    results/metrics/sentiment_lstm_metrics.csv  (only if the gate opens)

Inputs:
    data/processed/log_returns.csv     (RELIANCE column → next-day target)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from scipy.stats import pearsonr, spearmanr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# seed for reproducibility — VADER itself is deterministic, but keep the
# convention so the file matches the rest of the pipeline
np.random.seed(42)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
PROC_DIR = ROOT / "data" / "processed"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
for d in (PROC_DIR, MET_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_OUT = PROC_DIR / "sentiment_raw.csv"
SCORES_OUT = PROC_DIR / "sentiment_scores.csv"
CORR_OUT = MET_DIR / "sentiment_correlation.csv"
SCATTER_OUT = FIG_DIR / "sentiment_correlation.png"
TIMELINE_OUT = FIG_DIR / "sentiment_timeline.png"

HERO = "RELIANCE"
TICKER_NS = "RELIANCE.NS"

# Sprint 03 split boundaries — kept for the augmented-LSTM gate's overlap check.
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")

# Gate thresholds for the augmented LSTM: need at least 100 aligned daily
# rows and at least 60 of those falling inside train+val so there's enough
# supervised signal to fit a sequence model. yfinance's tiny window will
# almost always trip the first one.
GATE_MIN_ALIGNED = 100
GATE_MIN_OVERLAP = 60

ROLLING_WINDOW = 5

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Step 1 — fetch headlines (with cache)
# ---------------------------------------------------------------------------
def fetch_headlines(force_refresh: bool = False) -> pd.DataFrame:
    """Pull yfinance news for RELIANCE.NS and return a tidy DataFrame.

    yfinance only goes back a short window — we'll work with what we have
    and be upfront about this in the report.

    Schema observed in 2026-05 probe: each item is a dict with
    `id` and `content.{title,summary,pubDate,...}`. pubDate is ISO8601
    with a trailing Z (UTC). We extract title + summary + pubDate, and
    fall back gracefully if `summary` is missing on some items.
    """
    if RAW_OUT.exists() and not force_refresh:
        print(f"  Using cached headlines from {RAW_OUT}")
        df = pd.read_csv(RAW_OUT, parse_dates=["pubDate"])
        return df

    print(f"  Fetching yfinance news for {TICKER_NS}...")
    items = yf.Ticker(TICKER_NS).news or []
    print(f"  Got {len(items)} items from yfinance")

    rows = []
    for it in items:
        content = it.get("content") or {}
        title = (content.get("title") or "").strip()
        summary = (content.get("summary") or "").strip()
        pub_date_str = content.get("pubDate") or content.get("displayTime")
        if not title or not pub_date_str:
            continue
        try:
            # ISO8601 with Z — pandas parses this directly to UTC.
            pub_date = pd.to_datetime(pub_date_str, utc=True)
        except Exception:
            continue
        rows.append(
            {
                "id": it.get("id"),
                "pubDate": pub_date,
                "title": title,
                "summary": summary,
            }
        )

    if not rows:
        # Defensive — if both the network and the cache fail, exit cleanly
        # with a message rather than crashing on an empty groupby.
        raise SystemExit(
            "No headlines available (network returned empty and no cache). "
            "Re-run with network access; the snapshot will be persisted on first success."
        )

    df = pd.DataFrame(rows).sort_values("pubDate").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Step 2 — VADER scoring
# ---------------------------------------------------------------------------
def score_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER pos/neg/neu/compound columns to the headlines frame.

    We score `title + ". " + summary` rather than title-only — VADER
    handles longer text fine and the summaries often carry the actual
    sentiment-laden detail (e.g. "shares fell 3% as ..."). For headlines
    without a summary the score reduces to title-only by construction.
    """
    analyser = SentimentIntensityAnalyzer()
    scored: list[dict] = []
    for _, row in df.iterrows():
        text = row["title"]
        if isinstance(row.get("summary"), str) and row["summary"]:
            text = f"{row['title']}. {row['summary']}"
        s = analyser.polarity_scores(text)
        scored.append({**row.to_dict(), **{f"vader_{k}": v for k, v in s.items()}})
    return pd.DataFrame(scored)


# ---------------------------------------------------------------------------
# Step 3 — daily aggregation + 5-day rolling
# ---------------------------------------------------------------------------
def aggregate_daily(scored: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per IST publication date.

    Convert pubDate (UTC) to IST so a 19:00 UTC headline (00:30 next-day
    IST) is grouped against the next trading session — that lines up with
    how an Indian retail trader would actually see the news. Returns
    a DataFrame indexed by Date with columns daily_compound, n_headlines,
    rolling_compound (5-day, min_periods=1).
    """
    ist = scored.copy()
    ist["pubDate_ist"] = ist["pubDate"].dt.tz_convert("Asia/Kolkata")
    ist["Date"] = ist["pubDate_ist"].dt.date

    grouped = (
        ist.groupby("Date")
        .agg(daily_compound=("vader_compound", "mean"), n_headlines=("vader_compound", "size"))
        .sort_index()
    )
    grouped.index = pd.to_datetime(grouped.index)
    grouped["rolling_compound"] = (
        grouped["daily_compound"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    return grouped


# ---------------------------------------------------------------------------
# Step 4 — align to next-day log returns
# ---------------------------------------------------------------------------
def align_to_next_day_returns(daily: pd.DataFrame, log_returns: pd.Series) -> pd.DataFrame:
    """For each sentiment date d, attach the realised RELIANCE log return
    on the *next available trading day*. This handles weekends/holidays
    cleanly without imputing — if no trading day exists after d in the
    return series (e.g. d > last test date), drop that row.

    Resulting frame has daily_compound, rolling_compound, n_headlines,
    next_trading_date, next_day_logret. Indexed by sentiment date.
    """
    trading_dates = log_returns.index.sort_values()
    rows = []
    for d, row in daily.iterrows():
        # searchsorted with side="right" finds the first trading date strictly after d.
        i = trading_dates.searchsorted(d, side="right")
        if i >= len(trading_dates):
            continue
        next_d = trading_dates[i]
        rows.append(
            {
                "Date": d,
                "daily_compound": row["daily_compound"],
                "rolling_compound": row["rolling_compound"],
                "n_headlines": row["n_headlines"],
                "next_trading_date": next_d,
                "next_day_logret": float(log_returns.loc[next_d]),
            }
        )
    cols = ["daily_compound", "rolling_compound", "n_headlines",
            "next_trading_date", "next_day_logret"]
    if not rows:
        # Honest zero-overlap path. With the directive's locked date range
        # ending 2026-04-10 and yfinance only returning the last week or two
        # of news, every sentiment date sits after the last available return
        # — there's literally no next-day return to pair with. Return an
        # empty frame with the right schema so downstream code doesn't crash.
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols},
                            index=pd.DatetimeIndex([], name="Date"))
    return pd.DataFrame(rows).set_index("Date")


# ---------------------------------------------------------------------------
# Step 5 — correlation analysis + scatter
# ---------------------------------------------------------------------------
def correlation_analysis(aligned: pd.DataFrame) -> dict:
    n = len(aligned)
    if n < 3:
        # Pearson/Spearman are undefined below n=3; surface NaN rather than crash.
        print(f"  [warn] only {n} aligned points — correlation undefined")
        return {
            "n": n,
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
        }
    x = aligned["rolling_compound"].values
    y = aligned["next_day_logret"].values
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {
        "n": int(n),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
    }


def plot_timeline(daily: pd.DataFrame, save_to: Path) -> None:
    """Bars for daily compound + line for 5-day rolling, over the few dates
    we have. With only ~6 points the chart is sparse, but it's the most
    honest visual: a reader can see at a glance how few headlines we got
    and how the rolling smoother behaves on tiny windows."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["seagreen" if v >= 0 else "indianred" for v in daily["daily_compound"]]
    ax.bar(daily.index, daily["daily_compound"], color=colors, alpha=0.7,
           edgecolor="black", linewidth=0.5, width=0.7,
           label="Daily mean compound")
    ax.plot(daily.index, daily["rolling_compound"], color="navy", marker="o",
            linewidth=1.6, label=f"{ROLLING_WINDOW}-day rolling mean")
    for d, n in zip(daily.index, daily["n_headlines"]):
        ax.annotate(f"n={int(n)}", (d, daily.loc[d, "daily_compound"]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="grey")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title(
        "RELIANCE.NS — VADER compound sentiment timeline\n"
        f"({len(daily)} dates with at least one headline, "
        f"{daily.index.min().date()} → {daily.index.max().date()})"
    )
    ax.set_ylabel("VADER compound score")
    ax.set_xlabel("Date (IST)")
    ax.set_ylim(-1, 1)
    ax.legend(loc="lower left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


def plot_scatter(aligned: pd.DataFrame, stats: dict, save_to: Path,
                 daily: pd.DataFrame | None = None,
                 log_returns_end: pd.Timestamp | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    x = aligned["rolling_compound"].values
    y = aligned["next_day_logret"].values

    # When there's nothing to plot — which is the realistic case here, given
    # yfinance returns a few last-week headlines and the directive's locked
    # data range ends 2026-04-10 — drop a banner over the empty axes
    # explaining what happened. A blank chart with NaN in the title is more
    # confusing than informative.
    if stats["n"] == 0:
        msg_lines = [
            "Zero overlap between sentiment dates and price-data window.",
            f"yfinance returned {0 if daily is None else len(daily)} sentiment dates,",
        ]
        if daily is not None and len(daily) > 0:
            msg_lines.append(
                f"all in {daily.index.min().date()} → {daily.index.max().date()}"
            )
        if log_returns_end is not None:
            msg_lines.append(
                f"but log_returns.csv ends {log_returns_end.date()}."
            )
        msg_lines.append("Correlation undefined — see report for details.")
        ax.text(
            0.5, 0.5, "\n".join(msg_lines),
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow", edgecolor="grey"),
        )

    ax.scatter(x, y, color="navy", alpha=0.7, s=42, edgecolor="white", linewidth=0.6,
               label=f"n = {stats['n']}")

    # Thin OLS fit through the cloud — only meaningful if there are enough
    # points for the slope to be more than noise. polyfit needs >= 2.
    if stats["n"] >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, slope * xs + intercept, color="darkred", linestyle="--", linewidth=1.2,
                label=f"OLS fit: slope={slope:+.4f}")

    ax.axhline(0, color="grey", linewidth=0.6, alpha=0.6)
    ax.axvline(0, color="grey", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("5-day rolling VADER compound score (RELIANCE.NS headlines)")
    ax.set_ylabel("Next-day RELIANCE log return")
    pearson_txt = (
        f"Pearson r = {stats['pearson_r']:+.3f} (p = {stats['pearson_p']:.3f})"
        if not np.isnan(stats["pearson_r"])
        else "Pearson r = NaN"
    )
    spearman_txt = (
        f"Spearman ρ = {stats['spearman_r']:+.3f} (p = {stats['spearman_p']:.3f})"
        if not np.isnan(stats["spearman_r"])
        else "Spearman ρ = NaN"
    )
    ax.set_title(
        "Sentiment vs next-day return — RELIANCE\n"
        f"{pearson_txt}   |   {spearman_txt}"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


# ---------------------------------------------------------------------------
# Step 6 — gate check for the sentiment-augmented LSTM
# ---------------------------------------------------------------------------
def lstm_gate_check(aligned: pd.DataFrame) -> tuple[bool, dict]:
    """Decide whether we have enough data to bother training a 2-feature LSTM.

    The threshold is in two parts: total aligned rows and how many of
    those fall inside the existing train+val window. With only ≈10
    yfinance headlines from the last few days, both will fail — and that
    failure is exactly what the directive expects ("only include if data
    is sufficient").
    """
    n_aligned = len(aligned)
    in_trainval = aligned[aligned.index <= VAL_END]
    n_overlap = len(in_trainval)
    info = {
        "n_aligned": n_aligned,
        "n_overlap_trainval": n_overlap,
        "min_aligned_required": GATE_MIN_ALIGNED,
        "min_overlap_required": GATE_MIN_OVERLAP,
    }
    open_gate = n_aligned >= GATE_MIN_ALIGNED and n_overlap >= GATE_MIN_OVERLAP
    return open_gate, info


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Sprint 10 — sentiment correlation on RELIANCE.NS")
    print("=" * 70)

    force_refresh = os.environ.get("SENTIMENT_REFRESH", "").strip() not in ("", "0", "false", "False")

    print("\nStep 1 — fetch headlines")
    raw = fetch_headlines(force_refresh=force_refresh)
    print(f"  {len(raw)} headlines, "
          f"date range {raw['pubDate'].min().date()} → {raw['pubDate'].max().date()}")

    print("\nStep 2 — VADER scoring")
    scored = score_headlines(raw)
    scored.to_csv(RAW_OUT, index=False)
    print(f"  Saved → {RAW_OUT}  ({len(scored)} rows)")
    # Quick eyeball: most-positive and most-negative headlines (small n,
    # so we just print top-1 of each — enough to sanity-check VADER fired).
    if len(scored) >= 1:
        s_sorted = scored.sort_values("vader_compound")
        print("  Most negative:", f"({s_sorted.iloc[0]['vader_compound']:+.3f})",
              s_sorted.iloc[0]["title"][:90])
        print("  Most positive:", f"({s_sorted.iloc[-1]['vader_compound']:+.3f})",
              s_sorted.iloc[-1]["title"][:90])

    print("\nStep 3 — daily aggregate + 5-day rolling smooth")
    daily = aggregate_daily(scored)
    print(f"  {len(daily)} unique IST dates with at least one headline")
    print(daily.to_string())

    print("\nStep 4 — align to next-day RELIANCE log return")
    log_returns = (
        pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"])
        .sort_index()[HERO]
    )
    aligned = align_to_next_day_returns(daily, log_returns)
    print(f"  {len(aligned)} aligned (sentiment_date, next_trading_date) pairs")

    # Persist the aggregated + aligned scores frame. We keep the broader
    # `daily` (some sentiment dates may have no following trading day yet)
    # so a future re-run with newer returns can refill those rows.
    out_frame = daily.join(
        aligned[["next_trading_date", "next_day_logret"]], how="left"
    )
    out_frame.index.name = "Date"
    out_frame.to_csv(SCORES_OUT)
    print(f"  Saved → {SCORES_OUT}  (shape={out_frame.shape})")

    print("\nStep 5 — Pearson + Spearman + scatter plot")
    stats = correlation_analysis(aligned)
    print(f"  n = {stats['n']}")
    print(f"  Pearson  r = {stats['pearson_r']:+.4f}   p = {stats['pearson_p']:.4f}")
    print(f"  Spearman ρ = {stats['spearman_r']:+.4f}   p = {stats['spearman_p']:.4f}")

    if len(aligned) >= 1:
        date_range_start = aligned.index.min().date()
        date_range_end = aligned.index.max().date()
    else:
        date_range_start = None
        date_range_end = None
    corr_row = pd.DataFrame([
        {
            **stats,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "rolling_window": ROLLING_WINDOW,
        }
    ])
    corr_row.to_csv(CORR_OUT, index=False)
    print(f"  Saved → {CORR_OUT}")

    plot_scatter(aligned, stats, SCATTER_OUT,
                 daily=daily, log_returns_end=log_returns.index.max())
    plot_timeline(daily, TIMELINE_OUT)

    print("\nStep 6 — sentiment-augmented LSTM gate")
    open_gate, info = lstm_gate_check(aligned)
    print(f"  aligned rows: {info['n_aligned']:>4d}  (need ≥ {info['min_aligned_required']})")
    print(f"  rows in train+val (≤ {VAL_END.date()}): "
          f"{info['n_overlap_trainval']:>4d}  (need ≥ {info['min_overlap_required']})")
    if open_gate:
        # Future-proofing: if the gate ever opens (e.g. the directive is
        # later revised to use a wider news source), wire in the augmented
        # LSTM. For now this branch is intentionally a TODO that's loud
        # rather than silently doing the wrong thing on tiny data.
        print("  GATE OPEN — sentiment-augmented LSTM training would run here.")
        print("  [TODO] Implementation deferred; not exercised on this run.")
        print("  Skipping for this run; the correlation-only result above stands.")
    else:
        print("  GATE CLOSED — skipping sentiment-augmented LSTM. The yfinance window is")
        print("  too narrow and has no overlap with the 2016–2023 train/val span, so any")
        print("  trained model would be on a handful of rows from outside the existing")
        print("  scaffolding. Correlation-only result above stands. (Honest negative path,")
        print("  exactly as directive §10 anticipates.)")

    print("\nDone. Outputs in data/processed/, results/figures/, results/metrics/.")
    print("Sprint report → results/reports/10_sentiment.md")


if __name__ == "__main__":
    main()
