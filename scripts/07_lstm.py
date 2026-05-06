"""
07_lstm.py — Long Short-Term Memory network trained via Backpropagation
Through Time (BPTT). The flagship sequential model for this project; maps
to "Deep Neural Networks, Backpropagation Through Time" from the AI/ML
course (Bishop PRML Ch 5 + LSTM-gates intuition).

Architecture (per directive):

    (batch, 20, 1) → LSTM(64, return_seq=True) → Dropout(0.2)
                   → LSTM(32) [take last hidden state] → Dropout(0.2)
                   → Linear(32 → 1)

Adam (lr=1e-3) + MSE, max 100 epochs, batch 32, early stopping on val MSE
with patience 10. One model per ticker (univariate). Plus one optional
multivariate variant on RELIANCE that takes all 7 tickers' scaled log
returns as parallel features (input dim 7 instead of 1) — labelled
`LSTM_multivariate` so it row-stacks cleanly into Sprint 08's master
table without colliding with the univariate RELIANCE row.

Walk-forward one-step-ahead protocol on the 562-row test window matches
04_arima.py and 06_mlp.py exactly.

Inputs (from Sprint 03):
    data/processed/scaled/{TICKER}_scaled.npz   → train / val / test arrays
    data/processed/scaled/{TICKER}_scaler.joblib → fit on train, used to
                                                   inverse-transform predictions
    data/processed/log_returns.csv              → un-scaled targets for metrics
    data/processed/regime_labels.csv            → regime slicing
    data/ARIMA/aligned_prices.csv               → back-transform to price level

Outputs:
    results/figures/lstm_training_curve.png     → RELIANCE deep-dive only
    results/figures/lstm_reliance_forecast.png  → RELIANCE deep-dive only
    results/metrics/lstm_metrics.csv            → 32 rows × 6 cols
                                                  (7 univariate + 1 multivariate) × 4 metrics
    results/lstm_reliance.pt                    → univariate state_dict (Sprint 09 paper-trader)
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
# is enough to make this byte-deterministic across runs (verified on the MLP)
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=UserWarning)

# Pin to CPU for the same reasons as 06_mlp.py — the network is small, the
# data is ~1640 rows × 7 tickers, and CPU + seeded DataLoader gives
# byte-deterministic reruns out of the box.
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

VAL_END = pd.Timestamp("2023-12-31")  # same as Sprints 03–06

ARCH_LABEL_UNI = "LSTM(20→64,32)→1"
ARCH_LABEL_MULTI = "LSTM_multivariate"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# LSTM (Long Short-Term Memory) — a type of RNN that handles long-range dependencies
# Regular RNNs suffer from vanishing gradients over long sequences (the gradient shrinks
# as it's backpropagated through time — BPTT). LSTMs solve this with gating mechanisms
# (input gate, forget gate, output gate) that control what information flows through.
# This makes them much better suited for sequences than a flat MLP.
# Reference: Bishop PRML Ch 5, and also covered in the deep neural networks lecture.
# ---------------------------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, h1: int = 64, h2: int = 32, dropout: float = 0.2):
        super().__init__()
        # Two stacked LSTM layers. We don't use nn.LSTM(num_layers=2) directly
        # because we want a Dropout layer inserted between them on the
        # *sequence output* (PyTorch's built-in dropout only kicks in between
        # stacked layers when num_layers>1 and is applied to the inputs of
        # each layer except the first — same idea, but expressing it
        # explicitly here makes the directive's "return_sequences=True →
        # dropout → LSTM" wiring obvious).
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=h1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=h1, hidden_size=h2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, lookback, input_size)
        seq, _ = self.lstm1(x)          # (batch, lookback, h1) — full sequence
        seq = self.drop1(seq)
        out, _ = self.lstm2(seq)        # (batch, lookback, h2)
        last = out[:, -1, :]            # take the final-timestep hidden state
        last = self.drop2(last)
        return self.head(last).squeeze(-1)  # (batch,)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Window construction — same shape/stitching as 06_mlp.py
# (kept local; per-script standalone is the convention here)
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
    """Return dict with X/y arrays for train/val/test (numpy float32)."""
    npz = np.load(SCALED_DIR / f"{ticker}_scaled.npz")
    train, val, test = npz["train"], npz["val"], npz["test"]

    Xtr, ytr = make_xy_windows(train, LOOKBACK)

    val_seq = np.concatenate([train[-LOOKBACK:], val])
    Xv, yv = make_xy_windows(val_seq, LOOKBACK)
    assert len(yv) == len(val), f"{ticker}: val window count mismatch"

    test_seq = np.concatenate([val[-LOOKBACK:], test])
    Xte, yte = make_xy_windows(test_seq, LOOKBACK)
    assert len(yte) == len(test), f"{ticker}: test window count mismatch"

    return {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)}


def reshape_to_seq(X: np.ndarray) -> np.ndarray:
    # (N, 20) → (N, 20, 1) for the univariate LSTM
    return X.reshape(X.shape[0], X.shape[1], 1).astype(np.float32)


def build_multivariate_xy(target: str = HERO, tickers: list[str] = TICKERS) -> dict:
    """
    feeding in all tickers as features — the hypothesis is that NSEI and SENSEX
    movements might help predict RELIANCE, given how correlated they are
    (Sprint 01 found NSEI↔SENSEX corr 0.99, and both index the same broad
    market RELIANCE trades in).

    Returns the same dict shape as build_ticker_xy(), but X is (N, 20, 7)
    rather than (N, 20). The y target is the next-day *scaled* log return
    for `target` — column 0 in the stacked feature matrix by construction.
    """
    feature_order = [target] + [t for t in tickers if t != target]

    splits: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for t in feature_order:
        npz = np.load(SCALED_DIR / f"{t}_scaled.npz")
        for k in splits:
            splits[k].append(npz[k])

    # Sanity guard — Sprint 03 builds these from a single shared date index,
    # so all per-ticker splits should be the same length.
    for k in splits:
        lens = {len(a) for a in splits[k]}
        assert len(lens) == 1, f"multivariate {k}: ragged ticker lengths {lens}"

    mats = {k: np.stack(splits[k], axis=1).astype(np.float32) for k in splits}  # each: (T, 7)

    train_mat = mats["train"]
    val_seq = np.concatenate([train_mat[-LOOKBACK:], mats["val"]], axis=0)
    test_seq = np.concatenate([mats["val"][-LOOKBACK:], mats["test"]], axis=0)

    def windows(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # mat: (T, F); X: (T-LOOKBACK, LOOKBACK, F); y: (T-LOOKBACK,) using col 0.
        T, F = mat.shape
        if T <= LOOKBACK:
            return np.empty((0, LOOKBACK, F), dtype=np.float32), np.empty((0,), dtype=np.float32)
        # sliding_window_view over the first axis with shape (LOOKBACK, F)
        sw = np.lib.stride_tricks.sliding_window_view(mat, (LOOKBACK, F))  # (T-LOOKBACK+1, 1, LOOKBACK, F)
        sw = sw[:, 0, :, :]                                                 # (T-LOOKBACK+1, LOOKBACK, F)
        X = sw[:-1].copy().astype(np.float32)                               # (T-LOOKBACK, LOOKBACK, F)
        y = mat[LOOKBACK:, 0].copy().astype(np.float32)                     # (T-LOOKBACK,)
        return X, y

    Xtr, ytr = windows(train_mat)
    Xv, yv = windows(val_seq)
    Xte, yte = windows(test_seq)
    assert len(yv) == len(mats["val"]), "multivariate val window count mismatch"
    assert len(yte) == len(mats["test"]), "multivariate test window count mismatch"

    return {"train": (Xtr, ytr), "val": (Xv, yv), "test": (Xte, yte)}


# ---------------------------------------------------------------------------
# Training loop with early stopping (identical to 06_mlp.py)
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
# Metrics — same definitions as 04_arima.py / 06_mlp.py
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
def plot_training_curve(history: dict, save_to: Path, arch_label: str = ARCH_LABEL_UNI) -> None:
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
        f"RELIANCE — {arch_label} training curve\n"
        f"stopped at epoch {history['stopped_at_epoch']}, "
        f"best val MSE = {history['best_val']:.6f}"
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
    arch_label: str = ARCH_LABEL_UNI,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(test_dates, actual_price, color="navy", linewidth=1.2,
            label="Actual RELIANCE Adj Close")
    ax.plot(
        test_dates,
        pred_price,
        color="darkred",
        linewidth=1.0,
        linestyle="--",
        label=f"{arch_label} one-step-ahead",
    )
    ax.set_title(
        "RELIANCE — LSTM one-step-ahead forecast on test window\n"
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
# Per-model pipeline — train, predict, score
# ---------------------------------------------------------------------------
def _train_and_predict(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    Xte: np.ndarray,
    input_size: int,
    label: str,
) -> tuple[nn.Module, dict, np.ndarray]:
    """Shared train+predict block. Returns (model, history, scaled_test_preds)."""
    g = torch.Generator()
    g.manual_seed(42)
    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    val_ds = TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv))
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = LSTMRegressor(input_size=input_size).to(DEVICE)
    history = fit_with_early_stopping(model, train_loader, val_loader, label=label)

    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy().ravel()
    return model, history, preds_scaled


def run_ticker_univariate(
    ticker: str,
    log_returns: pd.DataFrame,
    regimes: pd.DataFrame,
) -> dict:
    print(f"\n  ── {ticker} (univariate) ──")
    xy = build_ticker_xy(ticker)
    Xtr, ytr = xy["train"]
    Xv, yv = xy["val"]
    Xte, yte_scaled = xy["test"]

    Xtr3, Xv3, Xte3 = reshape_to_seq(Xtr), reshape_to_seq(Xv), reshape_to_seq(Xte)
    print(
        f"    shapes: train X={Xtr3.shape} y={ytr.shape}; "
        f"val X={Xv3.shape} y={yv.shape}; test X={Xte3.shape} y={yte_scaled.shape}"
    )

    model, history, preds_scaled = _train_and_predict(
        Xtr3, ytr, Xv3, yv, Xte3, input_size=1, label=ticker
    )

    scaler = joblib.load(SCALED_DIR / f"{ticker}_scaler.joblib")
    preds_logret = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

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


def run_multivariate(log_returns: pd.DataFrame, regimes: pd.DataFrame) -> dict:
    print(f"\n  ── {HERO} (multivariate, 7-feature) ──")
    xy = build_multivariate_xy(target=HERO, tickers=TICKERS)
    Xtr, ytr = xy["train"]
    Xv, yv = xy["val"]
    Xte, yte_scaled = xy["test"]
    print(
        f"    shapes: train X={Xtr.shape} y={ytr.shape}; "
        f"val X={Xv.shape} y={yv.shape}; test X={Xte.shape} y={yte_scaled.shape}"
    )

    model, history, preds_scaled = _train_and_predict(
        Xtr, ytr, Xv, yv, Xte, input_size=Xtr.shape[2], label=f"{HERO}_multi"
    )

    # Inverse-transform with the RELIANCE scaler (target was the RELIANCE column).
    scaler = joblib.load(SCALED_DIR / f"{HERO}_scaler.joblib")
    preds_logret = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()

    test_dates = log_returns.loc[log_returns.index > VAL_END].index
    y_true_logret = log_returns.loc[test_dates, HERO].values
    assert len(y_true_logret) == len(preds_logret), \
        f"{HERO}_multi: prediction count mismatch"

    sliced = metrics_by_regime(test_dates, y_true_logret, preds_logret, regimes)
    print(
        f"    {HERO}_multi: RMSE={sliced['full_test']['rmse']:.5f}  "
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

    sample_uni = LSTMRegressor(input_size=1).to(DEVICE)
    sample_multi = LSTMRegressor(input_size=len(TICKERS)).to(DEVICE)
    print(
        f"\nLSTM univariate  {ARCH_LABEL_UNI}     params={count_params(sample_uni)}"
    )
    print(
        f"LSTM multivariate {ARCH_LABEL_MULTI}({len(TICKERS)}-feat) params={count_params(sample_multi)}"
    )

    print("\n" + "=" * 70)
    print(f"Training per-ticker LSTMs (max {MAX_EPOCHS} epochs, batch {BATCH_SIZE},")
    print(f"  Adam lr={LR}, early stopping patience={PATIENCE} on val MSE)")
    print("=" * 70)

    per_ticker: dict[str, dict] = {}
    for ticker in TICKERS:
        per_ticker[ticker] = run_ticker_univariate(ticker, log_returns, regimes)

    # ---- Multivariate variant (RELIANCE, 7 features) -----------------------
    print("\n" + "=" * 70)
    print("Multivariate LSTM on RELIANCE (all 7 tickers as input features)")
    print("=" * 70)
    multi_result = run_multivariate(log_returns, regimes)

    # ---- RELIANCE deep-dive figures (univariate model) ---------------------
    print("\n" + "=" * 70)
    print("RELIANCE deep-dive: training curve + forecast plot")
    print("=" * 70)
    plot_training_curve(
        per_ticker[HERO]["history"],
        FIG_DIR / "lstm_training_curve.png",
        arch_label=ARCH_LABEL_UNI,
    )

    test_dates = per_ticker[HERO]["test_dates"]
    preds_logret = per_ticker[HERO]["preds_logret"]
    actual_price_test = prices.loc[test_dates, HERO].values
    last_in_sample_price = float(prices.loc[prices.index <= VAL_END, HERO].iloc[-1])
    actual_prev = np.concatenate([[last_in_sample_price], actual_price_test[:-1]])
    pred_price = actual_prev * np.exp(preds_logret)
    plot_forecast(
        test_dates,
        actual_price_test,
        pred_price,
        FIG_DIR / "lstm_reliance_forecast.png",
        arch_label=ARCH_LABEL_UNI,
    )

    # ---- Save RELIANCE univariate state_dict for Sprint 09 -----------------
    rel_model_path = MODEL_DIR / "lstm_reliance.pt"
    torch.save(per_ticker[HERO]["model"].state_dict(), rel_model_path)
    print(f"\nSaved RELIANCE univariate state_dict → {rel_model_path}")

    # ---- Metrics CSV (schema matches mlp_metrics.csv, +4 multivariate rows)
    print("\n" + "=" * 70)
    print("Assembling lstm_metrics.csv")
    print("=" * 70)
    rows: list[dict] = []
    for ticker in TICKERS:
        sliced = per_ticker[ticker]["sliced"]
        for metric in ("rmse", "mae", "mape", "da"):
            rows.append(
                {
                    "ticker": ticker,
                    "order": ARCH_LABEL_UNI,
                    "metric": metric,
                    "full_test": sliced["full_test"][metric],
                    "regime_calm": sliced["regime_calm"][metric],
                    "regime_turbulent": sliced["regime_turbulent"][metric],
                }
            )
    # Multivariate rows for RELIANCE — distinct `order` so Sprint 08 can pick
    # it apart from the univariate RELIANCE row in a one-liner groupby.
    for metric in ("rmse", "mae", "mape", "da"):
        rows.append(
            {
                "ticker": HERO,
                "order": ARCH_LABEL_MULTI,
                "metric": metric,
                "full_test": multi_result["sliced"]["full_test"][metric],
                "regime_calm": multi_result["sliced"]["regime_calm"][metric],
                "regime_turbulent": multi_result["sliced"]["regime_turbulent"][metric],
            }
        )

    metrics_df = pd.DataFrame(rows)
    metrics_path = MET_DIR / "lstm_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}  (shape={metrics_df.shape})")
    print(metrics_df.to_string(index=False))

    # ---- Headline tables ---------------------------------------------------
    da_table = metrics_df[metrics_df["metric"] == "da"][["ticker", "order", "full_test"]]
    print("\nDirectional accuracy (full test):")
    for _, row in da_table.iterrows():
        print(f"  {row['ticker']:8s} [{row['order']:20s}] {row['full_test']:.3f}")

    print("\nTraining summary:")
    for ticker in TICKERS:
        h = per_ticker[ticker]["history"]
        print(
            f"  {ticker:8s} stopped_at={h['stopped_at_epoch']:3d}  "
            f"best_epoch={h['best_epoch']:3d}  best_val={h['best_val']:.6f}"
        )
    h = multi_result["history"]
    print(
        f"  {HERO}_multi stopped_at={h['stopped_at_epoch']:3d}  "
        f"best_epoch={h['best_epoch']:3d}  best_val={h['best_val']:.6f}"
    )

    # ---- LSTM vs MLP head-to-head DA (RELIANCE focus) ----------------------
    mlp_metrics_path = MET_DIR / "mlp_metrics.csv"
    if mlp_metrics_path.exists():
        mlp_df = pd.read_csv(mlp_metrics_path)
        mlp_da = mlp_df[mlp_df["metric"] == "da"].set_index("ticker")["full_test"]
        lstm_da_uni = (
            metrics_df[(metrics_df["metric"] == "da") & (metrics_df["order"] == ARCH_LABEL_UNI)]
            .set_index("ticker")["full_test"]
        )
        print("\nDA head-to-head (LSTM_uni vs MLP):")
        for ticker in TICKERS:
            delta = lstm_da_uni[ticker] - mlp_da[ticker]
            print(
                f"  {ticker:8s}  MLP={mlp_da[ticker]:.3f}  "
                f"LSTM_uni={lstm_da_uni[ticker]:.3f}  Δ={delta:+.3f}"
            )
        multi_da = multi_result["sliced"]["full_test"]["da"]
        print(
            f"\n  RELIANCE multi vs uni: uni={lstm_da_uni[HERO]:.3f}  "
            f"multi={multi_da:.3f}  Δ={multi_da - lstm_da_uni[HERO]:+.3f}"
        )

    print("\nDone. Outputs in results/figures/, results/metrics/, results/.")


if __name__ == "__main__":
    main()
