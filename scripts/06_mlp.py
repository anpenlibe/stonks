"""
06_mlp.py — Deep MLP via gradient descent and backpropagation. The first
neural-net baseline. Maps to Bishop PRML Ch 5 (forward pass, gradient
descent, backprop) from the AI/ML course.

Architecture (per directive): a 4-layer fully-connected feed-forward network
that takes a flat 20-day window of log returns and outputs a single
predicted next-day log return.

    20 → 64 (ReLU) → 32 (ReLU) → 16 (ReLU) → 1

Trained per-ticker with Adam + MSE, max 100 epochs, batch 32, early stopping
on val loss with patience 10. Walk-forward one-step-ahead on the test window
matches the protocol used by 04_arima.py so Sprint 08's benchmark stays
apples-to-apples (same 562 test predictions, same regime slicing schema).

Inputs (from Sprint 03):
    data/processed/scaled/{TICKER}_scaled.npz   → train / val / test arrays
    data/processed/scaled/{TICKER}_scaler.joblib → fit on train, used to
                                                   inverse-transform predictions
    data/processed/log_returns.csv              → un-scaled targets for metrics
    data/processed/regime_labels.csv            → regime slicing
    data/ARIMA/aligned_prices.csv               → back-transform to price level

Outputs:
    results/figures/mlp_training_curve.png      → RELIANCE deep-dive only
    results/figures/mlp_reliance_forecast.png   → RELIANCE deep-dive only
    results/metrics/mlp_metrics.csv             → 7 tickers × 4 metrics, long form
    results/mlp_reliance.pt                     → state_dict (Sprint 09 paper-trader)
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# seed for reproducibility — CPU + manual seeds + a generator-seeded DataLoader
# is enough to make this byte-deterministic across runs
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)

# CUDA is available on this box but the networks are tiny (~3.3k params) and
# the train sets are ~1640 rows × 7 tickers. CPU finishes faster end-to-end
# than CUDA initialisation overhead, and gives reproducible results out of
# the box. Pinning to CPU explicitly so a future GPU run doesn't subtly
# change the numbers.
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent.parent
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
REGIMES = ROOT / "data" / "processed" / "regime_labels.csv"
ALIGNED = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
SCALED_DIR = ROOT / "data" / "processed" / "scaled"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
MODEL_DIR = ROOT / "results"
for d in (FIG_DIR, MET_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]
HERO = "RELIANCE"

LOOKBACK = 20
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
LR = 1e-3

VAL_END = pd.Timestamp("2023-12-31")  # same as Sprints 03–05

ARCH_LABEL = "20→64→32→16→1"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# MLP architecture — as covered in class (Bishop Ch 5)
# 3 hidden layers felt like a reasonable depth — went deeper but it started
# overfitting on the training set
# ReLU is standard for regression networks, avoids vanishing gradients
# compared to sigmoid
# Adam is a gradient descent variant with per-parameter adaptive learning
# rates — converges faster than vanilla SGD
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, lookback: int = LOOKBACK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lookback, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback) → (batch, 1) → (batch,)
        return self.net(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Window construction — same shape as 03_preprocessing.make_windows but
# stitched from train+val+test slices so the first val/test windows have
# proper 20-day context
# ---------------------------------------------------------------------------
def make_xy_windows(arr: np.ndarray, lookback: int = LOOKBACK) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) <= lookback:
        return np.empty((0, lookback), dtype=np.float32), np.empty((0,), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(arr, lookback)
    X = windows[:-1].copy().astype(np.float32)
    y = arr[lookback:].copy().astype(np.float32)
    return X, y


def build_ticker_xy(ticker: str) -> dict:
    """Return a dict with X/y arrays for train, val, test (numpy float32)."""
    npz = np.load(SCALED_DIR / f"{ticker}_scaled.npz")
    train, val, test = npz["train"], npz["val"], npz["test"]

    # Train windows from train alone — first 20 train days are dropped
    Xtr, ytr = make_xy_windows(train, LOOKBACK)

    # Val windows: last LOOKBACK of train as context for the first val window.
    # Concatenate then slice so y aligns with val days (not train days).
    val_seq = np.concatenate([train[-LOOKBACK:], val])
    Xv, yv = make_xy_windows(val_seq, LOOKBACK)
    # Xv now has len(val) rows — sanity check
    assert len(yv) == len(val), f"{ticker}: val window count mismatch"

    # Test windows: same trick — last LOOKBACK of val as context for first test
    test_seq = np.concatenate([val[-LOOKBACK:], test])
    Xte, yte = make_xy_windows(test_seq, LOOKBACK)
    assert len(yte) == len(test), f"{ticker}: test window count mismatch"

    return {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)}


# ---------------------------------------------------------------------------
# Training loop with early stopping
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        # weight by batch size to get a true mean over the epoch (last batch
        # might be smaller)
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(total_n, 1)


def fit_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    lr: float = LR,
    label: str = "",
) -> dict:
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improve = 0

    # early stopping prevents overfitting — we stop when the model starts
    # memorising training data rather than learning patterns
    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = evaluate(model, val_loader, loss_fn)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"    {label} epoch {epoch:3d} | "
                f"train={train_loss:.6f}  val={val_loss:.6f}  "
                f"best_val={best_val:.6f}@{best_epoch}  patience={epochs_since_improve}"
            )

        if epochs_since_improve >= patience:
            print(
                f"    {label} early-stopping at epoch {epoch} "
                f"(best val={best_val:.6f} at epoch {best_epoch})"
            )
            break
    else:
        print(f"    {label} reached MAX_EPOCHS={max_epochs} without early stopping")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "stopped_at_epoch": len(train_losses),
    }


# ---------------------------------------------------------------------------
# Metrics — same definitions as 04_arima.py, reimplemented here to avoid the
# digit-prefixed-module-import dance
# ---------------------------------------------------------------------------
def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_pred) == np.sign(y_true)).mean()) if len(y_true) else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"rmse": np.nan, "mae": np.nan, "mape": np.nan, "da": np.nan}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "da": directional_accuracy(y_true, y_pred)}


def metrics_by_regime(
    test_dates: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_labels: pd.DataFrame,
) -> dict[str, dict]:
    full = compute_metrics(y_true, y_pred)
    aligned = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred}, index=test_dates
    ).join(regime_labels[["regime"]], how="inner")
    calm = aligned[aligned["regime"] == 0]
    turb = aligned[aligned["regime"] == 1]
    return {
        "full_test": full,
        "regime_calm": compute_metrics(calm["y_true"].values, calm["y_pred"].values),
        "regime_turbulent": compute_metrics(turb["y_true"].values, turb["y_pred"].values),
    }


# ---------------------------------------------------------------------------
# Plots — RELIANCE deep-dive only
# ---------------------------------------------------------------------------
def plot_training_curve(history: dict, save_to: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    epochs = np.arange(1, len(history["train_losses"]) + 1)
    ax.plot(epochs, history["train_losses"], color="navy", linewidth=1.4, label="train MSE")
    ax.plot(epochs, history["val_losses"], color="darkred", linewidth=1.4, label="val MSE")
    ax.axvline(
        history["best_epoch"],
        color="black",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label=f"best val @ epoch {history['best_epoch']}",
    )
    ax.set_title(
        f"RELIANCE — MLP {ARCH_LABEL} training curve\n"
        f"stopped at epoch {history['stopped_at_epoch']}, best val MSE = {history['best_val']:.6f}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled log-return space)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


def plot_forecast(
    test_dates: pd.DatetimeIndex,
    actual_price: np.ndarray,
    pred_price: np.ndarray,
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
        label=f"MLP {ARCH_LABEL} one-step-ahead",
    )
    ax.set_title(
        "RELIANCE — MLP one-step-ahead forecast on test window\n"
        "(predicted price = previous actual close × exp(predicted log return))"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close (₹)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


# ---------------------------------------------------------------------------
# Per-ticker pipeline — train, predict, score
# ---------------------------------------------------------------------------
def run_ticker(
    ticker: str,
    log_returns: pd.DataFrame,
    regimes: pd.DataFrame,
) -> dict:
    print(f"\n  ── {ticker} ──")
    xy = build_ticker_xy(ticker)
    Xtr, ytr = xy["train"]
    Xv, yv = xy["val"]
    Xte, yte_scaled = xy["test"]
    print(
        f"    shapes: train X={Xtr.shape} y={ytr.shape}; "
        f"val X={Xv.shape} y={yv.shape}; test X={Xte.shape} y={yte_scaled.shape}"
    )

    # Tensors + loaders. Deterministic shuffle via a generator-seeded DataLoader.
    g = torch.Generator()
    g.manual_seed(42)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = MLP(LOOKBACK).to(DEVICE)
    history = fit_with_early_stopping(
        model, train_loader, val_loader, label=ticker
    )

    # Test predictions in scaled space, then inverse-transform via the saved
    # train-only-fit scaler back to log-return space.
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy().ravel()

    scaler = joblib.load(SCALED_DIR / f"{ticker}_scaler.joblib")
    preds_logret = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    # Realised log returns straight from log_returns.csv (avoids round-tripping
    # through the scaler, which would only introduce floating-point noise).
    test_dates = log_returns.loc[log_returns.index > VAL_END].index
    y_true_logret = log_returns.loc[test_dates, ticker].values
    assert len(y_true_logret) == len(preds_logret), \
        f"{ticker}: prediction count mismatch with realised test log-returns"

    sliced = metrics_by_regime(test_dates, y_true_logret, preds_logret, regimes)
    print(
        f"    {ticker}: RMSE={sliced['full_test']['rmse']:.5f}  "
        f"MAE={sliced['full_test']['mae']:.5f}  "
        f"DA={sliced['full_test']['da']:.3f}"
    )

    return {
        "model": model,
        "history": history,
        "test_dates": test_dates,
        "y_true_logret": y_true_logret,
        "preds_logret": preds_logret,
        "sliced": sliced,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading log returns from {LOG_RET}")
    log_returns = pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"]).sort_index()
    print(f"  shape={log_returns.shape}")

    print(f"Loading regime labels from {REGIMES}")
    regimes = pd.read_csv(REGIMES, index_col="Date", parse_dates=["Date"]).sort_index()
    print(f"  {len(regimes)} labelled rows")

    print(f"Loading aligned prices from {ALIGNED} (for back-transform)")
    prices = pd.read_csv(ALIGNED, index_col="Date", parse_dates=["Date"]).sort_index()

    # Sanity-print the architecture once
    sample_model = MLP(LOOKBACK).to(DEVICE)
    print(f"\nMLP architecture: {ARCH_LABEL}  (params={count_params(sample_model)})")

    print("\n" + "=" * 70)
    print(f"Training per-ticker MLPs (max {MAX_EPOCHS} epochs, batch {BATCH_SIZE},")
    print(f"  Adam lr={LR}, early stopping patience={PATIENCE} on val MSE)")
    print("=" * 70)

    per_ticker: dict[str, dict] = {}
    for ticker in TICKERS:
        per_ticker[ticker] = run_ticker(ticker, log_returns, regimes)

    # ---- RELIANCE deep-dive figures ---------------------------------------
    print("\n" + "=" * 70)
    print("RELIANCE deep-dive: training curve + forecast plot")
    print("=" * 70)
    plot_training_curve(per_ticker[HERO]["history"], FIG_DIR / "mlp_training_curve.png")

    test_dates = per_ticker[HERO]["test_dates"]
    preds_logret = per_ticker[HERO]["preds_logret"]
    actual_price_test = prices.loc[test_dates, HERO].values
    last_in_sample_price = float(prices.loc[prices.index <= VAL_END, HERO].iloc[-1])
    actual_prev = np.concatenate([[last_in_sample_price], actual_price_test[:-1]])
    pred_price = actual_prev * np.exp(preds_logret)
    plot_forecast(test_dates, actual_price_test, pred_price, FIG_DIR / "mlp_reliance_forecast.png")

    # ---- Save RELIANCE state_dict for Sprint 09 ----------------------------
    rel_model_path = MODEL_DIR / "mlp_reliance.pt"
    torch.save(per_ticker[HERO]["model"].state_dict(), rel_model_path)
    print(f"\nSaved RELIANCE state_dict → {rel_model_path}")

    # ---- Metrics CSV (schema matches arima_metrics.csv) --------------------
    print("\n" + "=" * 70)
    print("Assembling mlp_metrics.csv")
    print("=" * 70)
    rows: list[dict] = []
    for ticker in TICKERS:
        sliced = per_ticker[ticker]["sliced"]
        for metric in ("rmse", "mae", "mape", "da"):
            rows.append(
                {
                    "ticker": ticker,
                    "order": ARCH_LABEL,  # column reused for the architecture label
                    "metric": metric,
                    "full_test": sliced["full_test"][metric],
                    "regime_calm": sliced["regime_calm"][metric],
                    "regime_turbulent": sliced["regime_turbulent"][metric],
                }
            )
    metrics_df = pd.DataFrame(rows)
    metrics_path = MET_DIR / "mlp_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}  (shape={metrics_df.shape})")
    print(metrics_df.to_string(index=False))

    # Headline DA table
    da_table = metrics_df[metrics_df["metric"] == "da"][["ticker", "full_test"]]
    print("\nDirectional accuracy by ticker (full test):")
    for _, row in da_table.iterrows():
        print(f"  {row['ticker']:8s} {row['full_test']:.3f}")

    # Per-ticker training summary
    print("\nTraining summary:")
    for ticker in TICKERS:
        h = per_ticker[ticker]["history"]
        print(
            f"  {ticker:8s} stopped_at={h['stopped_at_epoch']:3d}  "
            f"best_epoch={h['best_epoch']:3d}  best_val={h['best_val']:.6f}"
        )

    print("\nDone. Outputs in results/figures/, results/metrics/, results/.")


if __name__ == "__main__":
    main()
