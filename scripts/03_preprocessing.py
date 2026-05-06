"""
03_preprocessing.py — Build the leak-free, reusable inputs every downstream
script (04_arima.py through 10_sentiment.py) is going to lean on.

Concretely this script does five things:

1. Computes log returns from the aligned price frame and persists them as a
   single wide CSV at `data/processed/log_returns.csv`.
2. Carves out a strictly time-based train / val / test split. No shuffling,
   no random sampling — that would be a textbook leakage mistake on a time
   series (Brockwell & Davis Ch 5 hammer this in when they introduce
   forecast evaluation).
3. Joins the regime labels from Sprint 02 onto the returns frame so we have
   `regime_calm` and `regime_turbulent` subsets ready for script 08's
   regime-conditioned evaluation. Note: Sprint 02 already flagged that under
   the current k=2 labelling the test window (2024-01-01+) is 100% calm, so
   the "turbulent" subset effectively means COVID training-window days.
   We inherit that quirk honestly here and leave the resolution for script 08.
4. Fits a per-ticker MinMaxScaler on the **training set only** and persists
   each scaler with joblib, plus the scaled train/val/test arrays as npz
   files. The MLP and LSTM in 06/07 will load these directly.
5. Exposes a reusable `make_windows(series, lookback=20)` utility — the
   sliding-window converter that turns a 1-D return series into the
   (X, y) supervised-learning shape the neural nets expect.

Runnable standalone (`python scripts/03_preprocessing.py`) AND importable
as a module — the `if __name__ == "__main__":` guard at the bottom keeps
`make_windows` callable from 06_mlp.py / 07_lstm.py without re-running
the whole pipeline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# seed for reproducibility — nothing here is actually stochastic, but
# every script in this repo sets it at the top by convention
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
ALIGNED = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
REGIMES = ROOT / "data" / "processed" / "regime_labels.csv"
PROC_DIR = ROOT / "data" / "processed"
SCALED_DIR = PROC_DIR / "scaled"
MET_DIR = ROOT / "results" / "metrics"
PROC_DIR.mkdir(parents=True, exist_ok=True)
SCALED_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]

# Time-based split boundaries straight from the directive.
# Train ends 2022-12-31, validation is calendar 2023, test is 2024-01-01
# onwards. No shuffling — the index is the time axis.
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_END = pd.Timestamp("2023-12-31")

# Lookback used by the neural nets (20 trading days ≈ 1 month). Exposed as a
# module-level constant so 06_mlp.py and 07_lstm.py can import it.
LOOKBACK = 20


# ---------------------------------------------------------------------------
# Data loading — same idioms as 02_regime_detection.py for consistency
# ---------------------------------------------------------------------------
def load_aligned() -> pd.DataFrame:
    df = pd.read_csv(ALIGNED, index_col="Date", parse_dates=["Date"]).sort_index()
    return df[TICKERS]


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    # r_t = ln(P_t) - ln(P_{t-1}). Drops the first row where there's no prior
    # price to diff against. Same definition the EDA and regime scripts used —
    # keeping it consistent matters because downstream scripts assume identical
    # row counts and dates.
    return np.log(prices).diff().dropna(how="all")


# ---------------------------------------------------------------------------
# Time-based split — strictly chronological, no shuffling
# ---------------------------------------------------------------------------
def split_by_date(returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    idx = returns.index
    train = returns.loc[idx <= TRAIN_END]
    val = returns.loc[(idx > TRAIN_END) & (idx <= VAL_END)]
    test = returns.loc[idx > VAL_END]

    print("Time-based split:")
    for name, sub in [("train", train), ("val", val), ("test", test)]:
        print(
            f"  {name:5s}: {len(sub):4d} rows, "
            f"{sub.index.min().date()} → {sub.index.max().date()}"
        )
    # Sanity: no overlap, full coverage.
    assert len(train) + len(val) + len(test) == len(returns), "split rows don't sum to total"
    return train, val, test


# ---------------------------------------------------------------------------
# Regime subsets — joined from Sprint 02 labels
# ---------------------------------------------------------------------------
def regime_subsets(returns: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # Inner-join on Date so we only keep rows where both the return and a
    # regime label exist. The regime labels start ~60 days after the returns
    # because of the rolling-window warm-up in Sprint 02 — those early
    # un-labelled rows simply drop out of the regime subsets.
    joined = returns.join(labels[["regime"]], how="inner")
    calm = joined.loc[joined["regime"] == 0, returns.columns]
    turb = joined.loc[joined["regime"] == 1, returns.columns]

    print("\nRegime subsets (whole period):")
    print(f"  calm     : {len(calm):4d} rows")
    print(f"  turbulent: {len(turb):4d} rows")
    # Sprint 02 finding to keep front-of-mind for whoever reads this:
    # under k=2 the turbulent set is essentially the COVID window
    # (2020-03-20 → 2020-06-30). That's 67 days, ~2.8% of the labelled
    # period, and the test window has zero turbulent days. Script 08 will
    # have to decide whether to draw turbulent metrics from training-window
    # days or to re-label using percentile bands — flagged in the directive.
    return {"calm": calm, "turbulent": turb}


# ---------------------------------------------------------------------------
# Per-ticker scaling — fit on train only, persist scalers + scaled arrays
# ---------------------------------------------------------------------------
def fit_and_save_scalers(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    print("\nFitting MinMaxScalers per ticker (train-only fit)...")
    for ticker in TICKERS:
        # IMPORTANT: scaler is fit on train set only — fitting on the full
        # series would leak future information into the model, a classic
        # mistake (Bishop PRML Ch 1 warns about this when introducing
        # held-out evaluation). Even subtle leakage like fitting a scaler
        # on val/test data lets the model "see" future ranges.
        #
        # feature_range=(-1, 1) rather than the default (0, 1): log returns
        # are signed and centred near zero, so a symmetric range preserves
        # the directional signal instead of cramming negatives together.
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_arr = scaler.fit_transform(train[[ticker]].values).ravel()
        val_arr = scaler.transform(val[[ticker]].values).ravel()
        test_arr = scaler.transform(test[[ticker]].values).ravel()

        joblib.dump(scaler, SCALED_DIR / f"{ticker}_scaler.joblib")
        np.savez(
            SCALED_DIR / f"{ticker}_scaled.npz",
            train=train_arr,
            val=val_arr,
            test=test_arr,
        )
        print(
            f"  {ticker:8s}: train min/max={train_arr.min():+.3f}/{train_arr.max():+.3f}  "
            f"val min/max={val_arr.min():+.3f}/{val_arr.max():+.3f}  "
            f"test min/max={test_arr.min():+.3f}/{test_arr.max():+.3f}"
        )
    # If val/test min/max sit outside [-1, +1] that's actually a healthy
    # sign — it means a future return exceeded the train-period range,
    # which is honest and expected. Clamping would defeat the point.


# ---------------------------------------------------------------------------
# Sliding-window helper — used by 06_mlp.py and 07_lstm.py
# ---------------------------------------------------------------------------
def make_windows(series, lookback: int = LOOKBACK):
    """
    Convert a 1-D series into (X, y) for supervised learning.

    X[i] is the previous `lookback` observations; y[i] is the value at the
    prediction step. The first `lookback` rows are dropped because they
    don't have enough history. Returns numpy arrays.

    Shapes:
        X: (n_samples, lookback)
        y: (n_samples,)
    where n_samples = len(series) - lookback.

    The LSTM script will reshape X to (n_samples, lookback, 1); the MLP
    uses it flat as-is. Stride-tricks-based to avoid a Python loop.
    """
    arr = np.asarray(series, dtype=float).ravel()
    n_samples = len(arr) - lookback
    if n_samples <= 0:
        return np.empty((0, lookback)), np.empty((0,))
    # sliding_window_view gives all length-`lookback` windows of arr; we
    # then drop the last one because it has no corresponding y, and align y
    # to start at index `lookback`.
    windows = np.lib.stride_tricks.sliding_window_view(arr, lookback)
    X = windows[:-1].copy()
    y = arr[lookback:].copy()
    return X, y


# ---------------------------------------------------------------------------
# Splits summary table — what script 08 will consume
# ---------------------------------------------------------------------------
def write_splits_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []

    def add(name: str, sub: pd.DataFrame) -> None:
        if len(sub):
            rows.append(
                {
                    "split": name,
                    "start_date": sub.index.min().date(),
                    "end_date": sub.index.max().date(),
                    "n_rows": len(sub),
                }
            )
        else:
            rows.append({"split": name, "start_date": None, "end_date": None, "n_rows": 0})

    add("train", train)
    add("val", val)
    add("test", test)

    # Whole-period regime subsets, then split × regime cross-tab so script 08
    # can immediately see how many rows it'll be averaging metrics over.
    full = pd.concat([train, val, test])
    joined_full = full.join(labels[["regime"]], how="inner")

    add("regime_calm", joined_full.loc[joined_full["regime"] == 0])
    add("regime_turbulent", joined_full.loc[joined_full["regime"] == 1])

    for split_name, sub in [("train", train), ("val", val), ("test", test)]:
        joined = sub.join(labels[["regime"]], how="inner")
        add(f"{split_name}_calm", joined.loc[joined["regime"] == 0])
        add(f"{split_name}_turbulent", joined.loc[joined["regime"] == 1])

    summary = pd.DataFrame(rows)
    summary.to_csv(MET_DIR / "data_splits.csv", index=False)
    print(f"\nSaved splits summary → {MET_DIR / 'data_splits.csv'}")
    print(summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading aligned prices from {ALIGNED}")
    prices = load_aligned()
    print(
        f"  shape={prices.shape}, "
        f"range={prices.index.min().date()} → {prices.index.max().date()}"
    )

    returns = compute_log_returns(prices)
    out_returns = PROC_DIR / "log_returns.csv"
    returns.to_csv(out_returns)
    print(f"Saved log returns → {out_returns}  (shape={returns.shape})")

    print(f"Loading regime labels from {REGIMES}")
    labels = pd.read_csv(REGIMES, index_col="Date", parse_dates=["Date"]).sort_index()
    print(f"  {len(labels)} labelled rows from Sprint 02")

    train, val, test = split_by_date(returns)
    regime_subsets(returns, labels)  # printed-only side effects; in-memory dict not persisted
    fit_and_save_scalers(train, val, test)
    write_splits_summary(train, val, test, labels)

    # Quick smoke-test on make_windows so a fresh reader can see what shape
    # to expect when 06_mlp.py and 07_lstm.py call it.
    sample = train["RELIANCE"].values
    X, y = make_windows(sample, lookback=LOOKBACK)
    print(
        f"\nmake_windows smoke-test on RELIANCE train: "
        f"X.shape={X.shape}, y.shape={y.shape} (expected ({len(sample) - LOOKBACK}, {LOOKBACK}) and ({len(sample) - LOOKBACK},))"
    )

    print(f"\nDone. Outputs in {PROC_DIR} and {MET_DIR}.")


if __name__ == "__main__":
    main()
