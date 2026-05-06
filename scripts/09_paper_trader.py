"""
09_paper_trader.py — Paper trading on RELIANCE. Convert the one-step-ahead
log-return predictions from ARIMA / MLP / LSTM into BUY / HOLD / SELL signals
and simulate a portfolio from ₹1,00,000 over the test window
(2024-01-01 → 2026-04-10, 562 trading days), then compare against a
buy-and-hold benchmark.

Signal logic:
    pred log return > +0.5%  → BUY  (invest all cash at next open)
    pred log return < -0.5%  → SELL (liquidate all shares at next open)
    otherwise                → HOLD
Trades execute at that day's Open price. Long-only — a SELL with no shares
is recorded as the literal signal but doesn't move money. No transaction
costs.

Predictions are reconstructed deterministically from the same artefacts
Sprint 08 used:
- ARIMA: walk_forward_forecast from 04_arima.py with the order stored in
  arima_metrics.csv (RELIANCE → (0,0,0) on this dataset).
- MLP: forward-pass on results/mlp_reliance.pt (state_dict from Sprint 06).
- LSTM: forward-pass on results/lstm_reliance.pt (state_dict from Sprint 07,
  univariate input).
The three sibling helpers from 08_benchmark.py are imported via importlib
(filenames start with digits → plain import doesn't work) so we don't
re-derive them.

Outputs:
    results/trades/{arima,mlp,lstm,buy_and_hold}_trade_log.csv
    results/metrics/paper_trading_summary.csv
    results/figures/equity_curves.png

Inputs:
    data/processed/log_returns.csv               (562-day test DatetimeIndex)
    data/raw/RELIANCE.csv                        (Open prices for execution)
    data/processed/scaled/RELIANCE_{scaler.joblib,scaled.npz}
    results/{mlp,lstm}_reliance.pt
    results/metrics/arima_metrics.csv            (carries the RELIANCE order)
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

# seed for reproducibility — the heavy compute is upstream (Sprints 04/06/07);
# this script only does deterministic forward passes + a vectorised simulation,
# but pin seeds so reruns are byte-identical (same convention as 06/07/08)
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")  # matches Sprints 06/07/08

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
RAW_RELIANCE = ROOT / "data" / "raw" / "RELIANCE.csv"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR = ROOT / "results" / "figures"
TRADES_DIR = ROOT / "results" / "trades"
for d in (MET_DIR, FIG_DIR, TRADES_DIR):
    d.mkdir(parents=True, exist_ok=True)

HERO = "RELIANCE"
LOOKBACK = 20
VAL_END = pd.Timestamp("2023-12-31")  # same split as Sprints 03–08

# Trading parameters (directive §09)
START_CAPITAL = 100_000.0
BUY_THRESHOLD = 0.005   # +0.5% predicted log return → BUY
SELL_THRESHOLD = -0.005  # -0.5% predicted log return → SELL
RISK_FREE = 0.065        # 6.5% — approximate Indian T-bill yield (directive)
TRADING_DAYS = 252       # standard annualisation factor

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# importlib trick — same as 08. Filenames lead with digits, so we go through
# spec_from_file_location to lift the model classes + walk-forward helper +
# the three reliance_*_predictions helpers Sprint 08 already debugged.
# ---------------------------------------------------------------------------
def _load_module(label: str, filename: str):
    spec = importlib.util.spec_from_file_location(label, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Open-price loader — yfinance dumps a 3-line header on the raw CSVs (metric
# names, ticker repeats, then "Date,,,,,,"). align_prices.py already worked
# around this with skiprows=3 + explicit names; mirror that exactly.
# ---------------------------------------------------------------------------
def load_reliance_open(test_dates: pd.DatetimeIndex) -> pd.Series:
    df = pd.read_csv(
        RAW_RELIANCE,
        skiprows=3,
        header=None,
        names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
    )
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    open_px = df["Open"].reindex(test_dates).astype(float)
    if open_px.isna().any():
        missing = open_px[open_px.isna()].index.tolist()
        raise RuntimeError(f"Missing Open prices for {len(missing)} test dates: {missing[:5]}...")
    return open_px


# ---------------------------------------------------------------------------
# Simulator — long-only, fractional shares, no transaction costs
# ---------------------------------------------------------------------------
def simulate(preds: pd.Series, open_px: pd.Series, model_name: str) -> pd.DataFrame:
    """Step through the test window day by day. On each day:
        1. Read the model's predicted log return for that day.
        2. Map to BUY / SELL / HOLD via the ±0.5% thresholds.
        3. If BUY and we're in cash → invest everything at the day's open.
           If SELL and we hold shares → liquidate at the day's open.
           Otherwise → no transaction (HOLD, or already-positioned).
        4. Mark to market on the day's open price.

    Fractional shares: real exchanges don't allow it, but with ₹1,00,000 capital
    and RELIANCE around ₹1,300 we'd lose meaningful precision otherwise. The
    point of this sprint is comparing signal quality, not modelling lot-size
    quirks — flagging the assumption here.
    """
    cash = START_CAPITAL
    shares = 0.0
    rows = []
    for date, pred in preds.items():
        price = float(open_px.loc[date])

        if pred > BUY_THRESHOLD:
            signal = "BUY"
        elif pred < SELL_THRESHOLD:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Track whether the signal actually moved money — used downstream for
        # the n_trades count (a SELL while flat is a no-op, not a real trade).
        executed = False
        if signal == "BUY" and cash > 0:
            shares = cash / price
            cash = 0.0
            executed = True
        elif signal == "SELL" and shares > 0:
            cash = shares * price
            shares = 0.0
            executed = True

        portfolio_value = cash + shares * price
        rows.append({
            "date": date,
            "signal": signal,
            "executed": executed,
            "price": price,
            "shares": shares,
            "cash": cash,
            "portfolio_value": portfolio_value,
        })

    log = pd.DataFrame(rows).set_index("date")
    log.index.name = "date"
    return log


def simulate_buy_and_hold(open_px: pd.Series) -> pd.DataFrame:
    """Convert all cash to shares on day 1's open, then mark-to-market every
    day on that day's open. Same column schema as `simulate` so the four logs
    are interchangeable downstream."""
    first_price = float(open_px.iloc[0])
    shares = START_CAPITAL / first_price
    rows = []
    for i, (date, price) in enumerate(open_px.items()):
        signal = "BUY" if i == 0 else "HOLD"
        executed = i == 0
        portfolio_value = shares * float(price)
        rows.append({
            "date": date,
            "signal": signal,
            "executed": executed,
            "price": float(price),
            "shares": shares,
            "cash": 0.0,
            "portfolio_value": portfolio_value,
        })
    log = pd.DataFrame(rows).set_index("date")
    log.index.name = "date"
    return log


# ---------------------------------------------------------------------------
# Performance metrics — total / annualised return, Sharpe, max drawdown,
# n_trades, win rate. Each operates on one trade log.
# ---------------------------------------------------------------------------
def performance_summary(log: pd.DataFrame, model_name: str) -> dict:
    equity = log["portfolio_value"]
    n = len(equity)

    total_return_pct = (equity.iloc[-1] / START_CAPITAL - 1.0) * 100.0
    annualised_return_pct = (
        (equity.iloc[-1] / START_CAPITAL) ** (TRADING_DAYS / n) - 1.0
    ) * 100.0

    # Daily portfolio returns. On HOLD days where we're in cash, return is
    # zero; on HOLD days where we're long, it's the day's open-to-open price
    # change. .pct_change() handles both correctly.
    daily = equity.pct_change().dropna()
    rf_daily = RISK_FREE / TRADING_DAYS
    if daily.std(ddof=0) > 0:
        sharpe = (daily.mean() - rf_daily) / daily.std(ddof=0) * np.sqrt(TRADING_DAYS)
    else:
        # A model that never trades sits at start_capital all year → zero variance,
        # zero excess return → undefined Sharpe. Emit NaN + warn rather than divide.
        sharpe = float("nan")
        print(f"  [warn] {model_name}: zero return variance — Sharpe is undefined (model never traded)")

    # Max drawdown — peak-to-trough on the equity curve. Standard definition.
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown_pct = drawdown.min() * 100.0

    # n_trades counts only executed signals (a SELL while flat is recorded
    # in the log for transparency but not counted as a trade).
    n_trades = int(log["executed"].sum())

    # Win rate: pair each executed BUY with the next executed SELL → round trip.
    # Trade is profitable if exit_price > entry_price. A trailing BUY with no
    # closing SELL is "open" at end of test and excluded from the denominator.
    executed = log[log["executed"]]
    entries = []
    completed_round_trips = 0
    profitable_round_trips = 0
    for _, row in executed.iterrows():
        if row["signal"] == "BUY":
            entries.append(row["price"])
        elif row["signal"] == "SELL" and entries:
            entry_price = entries.pop(0)
            completed_round_trips += 1
            if row["price"] > entry_price:
                profitable_round_trips += 1
    if completed_round_trips > 0:
        win_rate_pct = 100.0 * profitable_round_trips / completed_round_trips
    else:
        win_rate_pct = float("nan")

    return {
        "model": model_name,
        "total_return_pct": round(total_return_pct, 4),
        "annualised_return_pct": round(annualised_return_pct, 4),
        "sharpe_ratio": round(sharpe, 4) if not np.isnan(sharpe) else np.nan,
        "max_drawdown_pct": round(max_drawdown_pct, 4),
        "n_trades": n_trades,
        "completed_round_trips": completed_round_trips,
        "win_rate_pct": round(win_rate_pct, 4) if not np.isnan(win_rate_pct) else np.nan,
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


# ---------------------------------------------------------------------------
# Equity curves figure
# ---------------------------------------------------------------------------
def plot_equity_curves(equity_curves: dict[str, pd.Series], save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    style = {
        "ARIMA": ("steelblue", "-"),
        "MLP": ("darkorange", "-"),
        "LSTM": ("darkred", "-"),
        "BuyAndHold": ("black", "--"),
    }
    for label, equity in equity_curves.items():
        c, ls = style.get(label, ("grey", "-"))
        ax.plot(equity.index, equity.values, color=c, linestyle=ls, linewidth=1.3, label=label)

    ax.axhline(START_CAPITAL, color="grey", linewidth=0.8, alpha=0.6)
    ax.set_title(
        "Paper trading equity curves — RELIANCE, ₹1,00,000 starting capital, ±0.5% threshold\n"
        "(test window 2024-01-01 → 2026-04-10 is 100% calm under Sprint 02 k=2 regime labelling)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (₹)")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Sprint 09 — Paper trader on RELIANCE")
    print("=" * 70)

    # Load shared scaffolding from 08 (which itself imports 04/06/07).
    # 08 already debugged the importlib pattern — reuse as-is.
    print("\nLoading sibling modules via importlib...")
    bench_mod = _load_module("bench_mod", "08_benchmark.py")

    print(f"Loading log returns from {LOG_RET}")
    log_returns = pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"]).sort_index()
    test_dates = log_returns.loc[log_returns.index > VAL_END].index
    print(f"  test window: {test_dates[0].date()} → {test_dates[-1].date()}  ({len(test_dates)} days)")
    assert len(test_dates) == 562, f"expected 562 test days, got {len(test_dates)}"

    print(f"Loading RELIANCE Open prices from {RAW_RELIANCE}")
    open_px = load_reliance_open(test_dates)
    print(f"  open[0] = ₹{open_px.iloc[0]:,.2f}, open[-1] = ₹{open_px.iloc[-1]:,.2f}")

    # ---- 1. Reconstruct one-step-ahead predictions ------------------------
    print("\n" + "-" * 70)
    print("Reconstructing RELIANCE one-step-ahead test predictions")
    print("-" * 70)
    master = bench_mod.load_master_metrics()
    arima_order = bench_mod._reliance_arima_order(master)
    arima_pred = bench_mod.reliance_arima_predictions(log_returns, arima_order)
    mlp_pred = bench_mod.reliance_mlp_predictions(log_returns)
    lstm_pred = bench_mod.reliance_lstm_predictions(log_returns)
    preds = {"ARIMA": arima_pred, "MLP": mlp_pred, "LSTM": lstm_pred}
    print("  Reconstructed:", {k: v.shape for k, v in preds.items()})

    # ---- 2. Simulate ------------------------------------------------------
    print("\n" + "-" * 70)
    print("Simulating portfolios")
    print("-" * 70)
    logs: dict[str, pd.DataFrame] = {}
    for name, pred in preds.items():
        print(f"  Running {name}...")
        logs[name] = simulate(pred, open_px, name)
    logs["BuyAndHold"] = simulate_buy_and_hold(open_px)

    # ---- 3. Save trade logs ------------------------------------------------
    print("\n" + "-" * 70)
    print("Trade logs")
    print("-" * 70)
    log_filenames = {
        "ARIMA": "arima_trade_log.csv",
        "MLP": "mlp_trade_log.csv",
        "LSTM": "lstm_trade_log.csv",
        "BuyAndHold": "buy_and_hold_trade_log.csv",
    }
    for name, fname in log_filenames.items():
        path = TRADES_DIR / fname
        logs[name].to_csv(path)
        n_exec = int(logs[name]["executed"].sum())
        print(f"  Saved {path}  rows={len(logs[name])}  executed_signals={n_exec}")

    # ---- 4. Performance summary -------------------------------------------
    print("\n" + "-" * 70)
    print("Performance summary")
    print("-" * 70)
    summary_rows = [performance_summary(logs[name], name) for name in
                    ["ARIMA", "MLP", "LSTM", "BuyAndHold"]]
    summary = pd.DataFrame(summary_rows)
    summary_path = MET_DIR / "paper_trading_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"\n  Saved → {summary_path}")

    # ---- 5. Equity curves figure ------------------------------------------
    print("\n" + "-" * 70)
    print("Equity curves figure")
    print("-" * 70)
    equity_curves = {name: logs[name]["portfolio_value"] for name in
                     ["ARIMA", "MLP", "LSTM", "BuyAndHold"]}
    plot_equity_curves(equity_curves, FIG_DIR / "equity_curves.png")

    # ---- 6. Sanity checks --------------------------------------------------
    bh_expected = START_CAPITAL * float(open_px.iloc[-1]) / float(open_px.iloc[0])
    bh_actual = float(logs["BuyAndHold"]["portfolio_value"].iloc[-1])
    assert abs(bh_actual - bh_expected) < 1e-6, (
        f"BuyAndHold final equity {bh_actual} doesn't match {bh_expected}"
    )
    print(f"\n  ✓ BuyAndHold final equity {bh_actual:,.2f} matches "
          f"START_CAPITAL × open[-1]/open[0] = {bh_expected:,.2f}")

    # ---- 7. Narrative ------------------------------------------------------
    print("\n" + "=" * 70)
    print("Narrative summary")
    print("=" * 70)
    bh_total = summary[summary["model"] == "BuyAndHold"]["total_return_pct"].iloc[0]
    print(f"\nBuy-and-hold total return over the 562-day test window: {bh_total:.2f}%.")
    beats = summary[summary["total_return_pct"] > bh_total]["model"].tolist()
    beats = [b for b in beats if b != "BuyAndHold"]
    if beats:
        print(f"  Models that beat buy-and-hold: {', '.join(beats)}.")
    else:
        print("  No model beats buy-and-hold. Honest result — beating B&H consistently is hard,")
        print("  and the Sprint 08 DM test already said the models can't statistically separate")
        print("  themselves from each other on RMSE. The signal threshold of ±0.5% is also a")
        print("  blunt instrument: with daily log-return σ ≈ 1.5% the threshold rarely fires.")
    print("\nQueued for Sprint 10 (sentiment): VADER on yfinance RELIANCE headlines.")


if __name__ == "__main__":
    main()
