from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "ARIMA"
OUT_DIR.mkdir(parents=True, exist_ok=True)

series = {}
for csv_path in sorted(RAW_DIR.glob("*.csv")):
    df = pd.read_csv(csv_path, skiprows=3, header=None,
                     names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    series[csv_path.stem] = df["Adj Close"]

aligned = pd.DataFrame(series).sort_index()
aligned = aligned.ffill().dropna(how="any")
aligned.index.name = "Date"

out_path = OUT_DIR / "aligned_prices.csv"
aligned.to_csv(out_path)
print(f"Wrote {out_path} with shape {aligned.shape}")
