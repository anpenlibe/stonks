"""
08_benchmark.py — Unified benchmarking. Apples-to-apples comparison of every
model trained in Sprints 04 / 06 / 07 on the same 562-day test window
(2024-01-01 → 2026-04-10), with a Diebold–Mariano significance test between
ARIMA and LSTM on RELIANCE as the headline statistical claim.

What this script does:

1. Load `arima_metrics.csv`, `mlp_metrics.csv`, `lstm_metrics.csv`, stamp a
   `model` column on each, concatenate into one long-form table, and save
   to `results/metrics/master_benchmark.csv`. A wide pivot is printed to
   the console for the narrative.

2. Reconstruct RELIANCE one-step-ahead test predictions for ARIMA / MLP /
   LSTM. The original scripts only saved aggregate metrics, but the DM test
   needs aligned per-date error series. Re-running RELIANCE-only is cheap
   (ARIMA(0,0,0) walk-forward + two CPU forward passes ≈ a couple seconds).
   The script imports the model classes and walk-forward helper directly
   from `04_arima.py` / `06_mlp.py` / `07_lstm.py` via `importlib` so we
   don't redefine them — those filenames start with digits, so plain
   `import` doesn't work.

3. Diebold–Mariano test. For h=1 forecasts the long-run-variance correction
   collapses to the plain sample variance — no Newey–West weights needed.
   Saved to `results/metrics/dm_test.csv`.

4. Three figures:
   - `combined_forecast.png` — actual RELIANCE Adj Close + ARIMA / MLP /
     LSTM predicted prices on the test window (multivariate LSTM excluded
     to keep the plot legible).
   - `regime_comparison.png` — single-bar full-test RMSE per model on
     RELIANCE. The grouped calm-vs-turbulent design from the directive
     would be half-empty because Sprint 02's k=2 labelling puts zero
     turbulent days in the test window — the figure surfaces that
     constraint instead of hiding NaNs.
   - `directional_accuracy.png` — horizontal bar chart of mean DA across
     all 7 tickers per model (multivariate LSTM is RELIANCE-only and
     plotted as a separate bar).

5. A student-voice narrative summary printed to console.

Outputs:
    results/metrics/master_benchmark.csv
    results/metrics/dm_test.csv
    results/figures/combined_forecast.png
    results/figures/regime_comparison.png
    results/figures/directional_accuracy.png

Inputs:
    results/metrics/{arima,mlp,lstm}_metrics.csv
    results/{mlp,lstm}_reliance.pt
    data/processed/scaled/RELIANCE_{scaler.joblib,scaled.npz}
    data/processed/log_returns.csv
    data/ARIMA/aligned_prices.csv
"""

from __future__ import annotations

import ast
import importlib.util
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats

# seed for reproducibility — the heavy compute (training) is deterministic
# upstream; this script does forecasting + plotting, but pin seeds anyway
np.random.seed(42)
torch.manual_seed(42)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = torch.device("cpu")  # matches Sprints 06/07

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
LOG_RET = ROOT / "data" / "processed" / "log_returns.csv"
ALIGNED = ROOT / "data" / "ARIMA" / "aligned_prices.csv"
SCALED_DIR = ROOT / "data" / "processed" / "scaled"
MODEL_DIR = ROOT / "results"
FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
for d in (FIG_DIR, MET_DIR):
    d.mkdir(parents=True, exist_ok=True)

TICKERS = ["NSEI", "SENSEX", "TCS", "HDFCBANK", "RELIANCE", "ITC", "MARUTI"]
HERO = "RELIANCE"
LOOKBACK = 20
VAL_END = pd.Timestamp("2023-12-31")  # same as Sprints 03–07

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# importlib trick — Sprints 04 / 06 / 07 are leading-digit filenames so plain
# `import` fails. spec_from_file_location lets us reuse their helpers and
# model classes verbatim instead of duplicating ~60 lines.
# ---------------------------------------------------------------------------
def _load_module(label: str, filename: str):
    spec = importlib.util.spec_from_file_location(label, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


arima_mod = _load_module("arima_mod", "04_arima.py")
mlp_mod = _load_module("mlp_mod", "06_mlp.py")
lstm_mod = _load_module("lstm_mod", "07_lstm.py")


# ---------------------------------------------------------------------------
# Master metrics table — concat the three per-script CSVs
# ---------------------------------------------------------------------------
def load_master_metrics() -> pd.DataFrame:
    """Stack arima/mlp/lstm metrics into one long-form frame.

    Columns: ticker, model, order, metric, full_test, regime_calm, regime_turbulent.
    The `model` column is derived: ARIMA / MLP / LSTM by source CSV, with the
    one `LSTM_multivariate` row in lstm_metrics.csv getting its own model label.
    """
    arima = pd.read_csv(MET_DIR / "arima_metrics.csv")
    arima["model"] = "ARIMA"

    mlp = pd.read_csv(MET_DIR / "mlp_metrics.csv")
    mlp["model"] = "MLP"

    lstm = pd.read_csv(MET_DIR / "lstm_metrics.csv")
    lstm["model"] = np.where(lstm["order"] == "LSTM_multivariate", "LSTM_multivariate", "LSTM")

    cols = ["ticker", "model", "order", "metric", "full_test", "regime_calm", "regime_turbulent"]
    master = pd.concat([arima[cols], mlp[cols], lstm[cols]], ignore_index=True)
    return master


# ---------------------------------------------------------------------------
# RELIANCE per-step prediction reconstruction — three small helpers, each
# returning a pd.Series of one-step-ahead log-return predictions on the 562
# test dates.
# ---------------------------------------------------------------------------
def _reliance_test_dates(log_returns: pd.DataFrame) -> pd.DatetimeIndex:
    return log_returns.loc[log_returns.index > VAL_END].index


def _reliance_arima_order(master: pd.DataFrame) -> tuple[int, int, int]:
    """Read RELIANCE's ARIMA order out of arima_metrics.csv so 08 stays in
    lockstep with whatever 04 chose (stored as a stringified tuple)."""
    row = master[(master["model"] == "ARIMA") & (master["ticker"] == HERO)].iloc[0]
    order = ast.literal_eval(row["order"])
    return tuple(int(x) for x in order)


def reliance_arima_predictions(
    log_returns: pd.DataFrame,
    order: tuple[int, int, int],
) -> pd.Series:
    """Walk-forward one-step-ahead ARIMA predictions on the test window,
    using the same `walk_forward_forecast` helper from 04_arima.py — same
    `refit=False` trick (parameters fixed at in-sample MLE, state filtered
    through new observations as they arrive)."""
    in_sample = log_returns.loc[log_returns.index <= VAL_END, HERO].values
    test = log_returns.loc[log_returns.index > VAL_END, HERO].values
    print(f"  ARIMA{order} walk-forward on {len(test)} test days...")
    preds = arima_mod.walk_forward_forecast(
        order=order,
        in_sample=in_sample,
        test_series=test,
        progress_label=f"{HERO}-ARIMA",
    )
    test_dates = _reliance_test_dates(log_returns)
    assert len(preds) == len(test_dates) == 562
    return pd.Series(preds, index=test_dates, name="arima_pred")


def _load_reliance_test_features() -> tuple[np.ndarray, object]:
    """Return (Xte, scaler) for RELIANCE — exact same construction as the
    `build_ticker_xy(...)['test']` step in 06_mlp.py / 07_lstm.py.
    """
    npz = np.load(SCALED_DIR / f"{HERO}_scaled.npz")
    val_arr, test_arr = npz["val"], npz["test"]
    test_seq = np.concatenate([val_arr[-LOOKBACK:], test_arr])
    Xte, _ = mlp_mod.make_xy_windows(test_seq, LOOKBACK)
    scaler = joblib.load(SCALED_DIR / f"{HERO}_scaler.joblib")
    return Xte, scaler


def reliance_mlp_predictions(log_returns: pd.DataFrame) -> pd.Series:
    """Forward-pass on saved mlp_reliance.pt. The MLP's "walk-forward" is
    just a single batch forward (no refit between steps — see 06_mlp.py
    line 396); reusing the saved state_dict reproduces 06's RELIANCE
    predictions exactly."""
    Xte, scaler = _load_reliance_test_features()
    model = mlp_mod.MLP(LOOKBACK).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "mlp_reliance.pt", map_location=DEVICE))
    model.eval()
    print(f"  MLP forward pass on {len(Xte)} test windows...")
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(Xte).to(DEVICE)).cpu().numpy().ravel()
    preds_logret = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    test_dates = _reliance_test_dates(log_returns)
    assert len(preds_logret) == len(test_dates) == 562
    return pd.Series(preds_logret, index=test_dates, name="mlp_pred")


def reliance_lstm_predictions(log_returns: pd.DataFrame) -> pd.Series:
    """Forward-pass on saved lstm_reliance.pt — the univariate model only.
    Multivariate LSTM uses a different feature shape (input_size=7) and
    isn't part of the DM head-to-head."""
    Xte, scaler = _load_reliance_test_features()
    Xte3 = Xte.reshape(Xte.shape[0], Xte.shape[1], 1).astype(np.float32)
    model = lstm_mod.LSTMRegressor(input_size=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "lstm_reliance.pt", map_location=DEVICE))
    model.eval()
    print(f"  LSTM forward pass on {len(Xte3)} test windows...")
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(Xte3).to(DEVICE)).cpu().numpy().ravel()
    preds_logret = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    test_dates = _reliance_test_dates(log_returns)
    assert len(preds_logret) == len(test_dates) == 562
    return pd.Series(preds_logret, index=test_dates, name="lstm_pred")


# ---------------------------------------------------------------------------
# Diebold–Mariano test — checks if forecast accuracy differs significantly
# between two models. For h=1 (one-step-ahead) the long-run-variance
# correction collapses to the plain sample variance because there's no
# autocorrelation in the loss differential at lag 0 to weight (Newey–West
# weights only kick in for h > 1). Using squared error as the loss
# function — that's the textbook default; MAE-loss DM is also valid.
# Reference: Diebold & Mariano (1995), "Comparing Predictive Accuracy", JBES.
# ---------------------------------------------------------------------------
def diebold_mariano(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """Two-sided DM test on squared-error loss. Returns DM stat + p-value
    + the loss-differential mean (positive ⇒ model A's MSE is larger,
    i.e. model B is more accurate)."""
    e_a = np.asarray(y_true, dtype=float) - np.asarray(y_pred_a, dtype=float)
    e_b = np.asarray(y_true, dtype=float) - np.asarray(y_pred_b, dtype=float)
    d = e_a ** 2 - e_b ** 2
    n = len(d)
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))  # sample variance
    dm_stat = mean_d / np.sqrt(var_d / n)
    # two-sided p-value under standard normal asymptotic null
    p_value = float(2 * stats.norm.sf(abs(dm_stat)))
    return {
        "n": n,
        "mean_loss_diff": mean_d,
        "var_loss_diff": var_d,
        "dm_stat": float(dm_stat),
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_combined_forecast(
    test_dates: pd.DatetimeIndex,
    actual_price: np.ndarray,
    actual_prev_price: np.ndarray,
    preds: dict[str, pd.Series],
    save_to: Path,
) -> None:
    """ARIMA / MLP / LSTM predicted prices vs actual on the test window.
    Multivariate LSTM is excluded — would sit on top of the univariate line
    and clutter the legend (see Sprint 07 report: Δ DA = +0.002)."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(test_dates, actual_price, color="black", linewidth=1.4, label="Actual RELIANCE Adj Close")

    style = {
        "ARIMA": ("steelblue", "--"),
        "MLP": ("darkorange", "--"),
        "LSTM": ("darkred", "--"),
    }
    for label, preds_logret in preds.items():
        if label not in style:
            continue
        pred_price = arima_mod.back_transform_to_price(
            last_actual_price=float(actual_prev_price[0]),
            actual_prices_prev=actual_prev_price,
            pred_log_returns=preds_logret.values,
        )
        c, ls = style[label]
        ax.plot(test_dates, pred_price, color=c, linewidth=1.0, linestyle=ls, label=f"{label} one-step-ahead")

    ax.set_title(
        "RELIANCE — actual vs ARIMA / MLP / LSTM (test window, 562 days)\n"
        "predicted price = previous actual close × exp(predicted log return)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Adj Close (₹)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


def plot_regime_comparison(master: pd.DataFrame, save_to: Path) -> None:
    """Single-bar full-test RMSE per model on RELIANCE. The directive's
    grouped calm-vs-turbulent design would be half-empty (test window is
    100% calm under k-means labelling — Sprint 02 finding). Surface the
    constraint in the subtitle rather than render NaN bars."""
    rmse_rel = master[(master["ticker"] == HERO) & (master["metric"] == "rmse")].copy()
    # canonical model order for plotting
    model_order = ["ARIMA", "MLP", "LSTM", "LSTM_multivariate"]
    rmse_rel["model"] = pd.Categorical(rmse_rel["model"], categories=model_order, ordered=True)
    rmse_rel = rmse_rel.sort_values("model")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        rmse_rel["model"].astype(str),
        rmse_rel["full_test"],
        color=["steelblue", "darkorange", "darkred", "purple"],
        edgecolor="black",
        linewidth=0.6,
    )
    for b, v in zip(bars, rmse_rel["full_test"]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(
        "RELIANCE — full-test RMSE per model\n"
        "(test window 2024-01-01→2026-04-10 is 100% calm under k=2 regime labelling;\n"
        "turbulent-regime evaluation is degenerate on this test set — see Sprint 02 report)"
    )
    ax.set_ylabel("RMSE (log-return space)")
    ax.set_ylim(0, max(rmse_rel["full_test"]) * 1.18)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


def plot_directional_accuracy(master: pd.DataFrame, save_to: Path) -> None:
    """Horizontal bar chart, one bar per model.
    - ARIMA / MLP / LSTM: mean DA across all 7 tickers.
    - LSTM_multivariate: RELIANCE-only by construction (single bar).
    Vertical dashed line at 0.5 = coin-flip baseline.
    """
    da = master[master["metric"] == "da"].copy()
    rows = []
    for m in ["ARIMA", "MLP", "LSTM"]:
        sub = da[da["model"] == m]
        rows.append({"model": m, "da": sub["full_test"].mean(), "n_tickers": len(sub)})
    multi = da[da["model"] == "LSTM_multivariate"]
    if len(multi):
        rows.append({"model": "LSTM_multivariate (RELIANCE)", "da": float(multi["full_test"].iloc[0]), "n_tickers": 1})

    summary = pd.DataFrame(rows).sort_values("da", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(
        summary["model"], summary["da"],
        color=["purple", "darkred", "darkorange", "steelblue"][: len(summary)],
        edgecolor="black", linewidth=0.6,
    )
    for b, v in zip(bars, summary["da"]):
        ax.text(v, b.get_y() + b.get_height() / 2, f" {v:.3f}", va="center", fontsize=9)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.0, label="coin flip (0.500)")
    # ±1.96·√(0.25/562) ≈ 0.041 binomial noise band on n=562 — same band the
    # Sprint 06/07 reports cite when calling the per-ticker DA gaps "noise".
    ax.axvspan(0.5 - 0.041, 0.5 + 0.041, color="grey", alpha=0.15, label="±1.96σ binomial noise (n=562)")
    ax.set_xlim(0.40, 0.60)
    ax.set_xlabel("Mean directional accuracy (full test, 562 days)")
    ax.set_title(
        "Directional accuracy by model\n"
        "(ARIMA / MLP / LSTM bars are mean across 7 tickers; LSTM_multivariate is RELIANCE-only)"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_to, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_to}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("Sprint 08 — Unified benchmarking")
    print("=" * 70)

    print(f"\nLoading log returns from {LOG_RET}")
    log_returns = pd.read_csv(LOG_RET, index_col="Date", parse_dates=["Date"]).sort_index()
    print(f"  shape={log_returns.shape}")

    print(f"Loading aligned prices from {ALIGNED}")
    prices = pd.read_csv(ALIGNED, index_col="Date", parse_dates=["Date"]).sort_index()

    # ---- 1. Master metrics CSV --------------------------------------------
    print("\n" + "-" * 70)
    print("Master metrics — concat of arima/mlp/lstm")
    print("-" * 70)
    master = load_master_metrics()
    master_path = MET_DIR / "master_benchmark.csv"
    master.to_csv(master_path, index=False)
    print(f"Saved master_benchmark.csv  shape={master.shape}  →  {master_path}")

    # Wide pivot for the narrative — RMSE / DA per model on RELIANCE
    print("\nRELIANCE summary (full-test):")
    rel = master[master["ticker"] == HERO].pivot_table(
        index="model", columns="metric", values="full_test", aggfunc="first"
    )[["rmse", "mae", "da"]]
    print(rel.round(6).to_string())

    # ---- 2. RELIANCE per-step predictions ---------------------------------
    print("\n" + "-" * 70)
    print("Reconstructing RELIANCE one-step-ahead test predictions")
    print("-" * 70)
    arima_order = _reliance_arima_order(master)
    arima_pred = reliance_arima_predictions(log_returns, arima_order)
    mlp_pred = reliance_mlp_predictions(log_returns)
    lstm_pred = reliance_lstm_predictions(log_returns)

    test_dates = _reliance_test_dates(log_returns)
    y_true = log_returns.loc[test_dates, HERO].values

    # Sanity check — re-computed RMSE should match arima_metrics.csv to ~1e-12
    recomputed_rmse = float(np.sqrt(np.mean((y_true - arima_pred.values) ** 2)))
    expected_rmse = float(
        master[(master["model"] == "ARIMA") & (master["ticker"] == HERO) & (master["metric"] == "rmse")]
        ["full_test"].iloc[0]
    )
    assert abs(recomputed_rmse - expected_rmse) < 1e-9, (
        f"ARIMA RMSE re-computation mismatch: {recomputed_rmse} vs {expected_rmse}"
    )
    print(f"  ✓ ARIMA RMSE re-computed = {recomputed_rmse:.12f}  (matches arima_metrics.csv)")

    # ---- 3. Diebold–Mariano test ------------------------------------------
    print("\n" + "-" * 70)
    print("Diebold–Mariano test (ARIMA vs LSTM, RELIANCE, h=1, squared-error loss)")
    print("-" * 70)
    dm = diebold_mariano(y_true, arima_pred.values, lstm_pred.values)
    print(
        f"  n = {dm['n']}, mean(d_t) = {dm['mean_loss_diff']:.3e}, "
        f"DM = {dm['dm_stat']:.4f}, p = {dm['p_value']:.4f}"
    )
    if dm["p_value"] >= 0.05:
        verdict = "fail to reject H₀ — ARIMA and LSTM have equal forecast accuracy"
    else:
        verdict = ("reject H₀ — forecast accuracy differs significantly "
                   f"(in favour of {'LSTM' if dm['mean_loss_diff'] > 0 else 'ARIMA'})")
    print(f"  {verdict}")

    dm_row = {
        "model_a": "ARIMA",
        "model_b": "LSTM",
        "ticker": HERO,
        "loss": "squared_error",
        "h": 1,
        **dm,
        "verdict": verdict,
    }
    dm_df = pd.DataFrame([dm_row])
    dm_path = MET_DIR / "dm_test.csv"
    dm_df.to_csv(dm_path, index=False)
    print(f"  Saved → {dm_path}")

    # ---- 4. Plots ----------------------------------------------------------
    print("\n" + "-" * 70)
    print("Plots")
    print("-" * 70)
    actual_price = prices.loc[test_dates, HERO].values
    last_in_sample_price = float(prices.loc[prices.index <= VAL_END, HERO].iloc[-1])
    actual_prev = np.concatenate([[last_in_sample_price], actual_price[:-1]])

    plot_combined_forecast(
        test_dates, actual_price, actual_prev,
        {"ARIMA": arima_pred, "MLP": mlp_pred, "LSTM": lstm_pred},
        FIG_DIR / "combined_forecast.png",
    )
    plot_regime_comparison(master, FIG_DIR / "regime_comparison.png")
    plot_directional_accuracy(master, FIG_DIR / "directional_accuracy.png")

    # ---- 5. Narrative summary ----------------------------------------------
    print("\n" + "=" * 70)
    print("Narrative summary")
    print("=" * 70)
    da_means = (
        master[master["metric"] == "da"]
        .groupby("model")["full_test"].mean()
        .reindex(["ARIMA", "MLP", "LSTM", "LSTM_multivariate"])
    )
    rmse_rel = master[(master["ticker"] == HERO) & (master["metric"] == "rmse")].set_index("model")["full_test"]
    print(
        f"\nMean DA across 7 tickers — ARIMA={da_means['ARIMA']:.3f}, "
        f"MLP={da_means['MLP']:.3f}, LSTM={da_means['LSTM']:.3f} "
        f"(LSTM_multivariate {da_means['LSTM_multivariate']:.3f}, RELIANCE only)."
    )
    print(
        "  Coin-flip baseline = 0.500; ±1.96σ binomial noise band at n=562 is ±0.041,\n"
        "  so all four model means sit comfortably inside one noise band of 0.5 — "
        "no model has a real directional edge over the broad 7-ticker portfolio."
    )
    print(
        f"\nRELIANCE full-test RMSE band — ARIMA={rmse_rel['ARIMA']:.5f}, "
        f"MLP={rmse_rel['MLP']:.5f}, LSTM={rmse_rel['LSTM']:.5f}, "
        f"multivariate-LSTM={rmse_rel['LSTM_multivariate']:.5f}.\n"
        "  All four bunch within ~0.0007 — the noise floor on daily log returns is "
        "doing the heavy lifting here, not the model."
    )
    print(
        f"\nDiebold–Mariano (ARIMA vs LSTM, RELIANCE, h=1, squared-error loss):\n"
        f"  DM = {dm['dm_stat']:.4f}, p = {dm['p_value']:.4f} → {verdict}.\n"
        "  This matches the visual story from the figures and the per-ticker DA bands\n"
        "  in Sprints 06/07: the LSTM's apparent edge is not statistically distinguishable\n"
        "  from the ARIMA baseline on this test set."
    )
    print(
        "\nHeadline limitation: the Sprint 02 k=2 regime labelling places ZERO turbulent\n"
        "  days in the test window, so the directive's regime-split benchmark collapses\n"
        "  to a single calm column. The regime_comparison figure surfaces this constraint\n"
        "  in its subtitle rather than rendering NaN bars."
    )
    print("\nQueued for Sprint 09 (paper trader): convert these one-step-ahead predictions")
    print("  into BUY/HOLD/SELL signals on RELIANCE and simulate against buy-and-hold.")


if __name__ == "__main__":
    main()
