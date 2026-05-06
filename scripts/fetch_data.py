import yfinance as yf
import os

OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)

symbols = {
    "Nifty50": "^NSEI",
    "Sensex": "^BSESN",
    "TCS": "TCS.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "Reliance": "RELIANCE.NS",
    "ITC": "ITC.NS",
    "Maruti": "MARUTI.NS",
}

for name, sym in symbols.items():
    df = yf.download(sym, period="10y", interval="1d", progress=False, auto_adjust=False)
    start = df.index.min().date() if len(df) else "NA"
    end = df.index.max().date() if len(df) else "NA"
    print(f"{name} ({sym}): rows={len(df)}, start={start}, end={end}")
    df.to_csv(os.path.join(OUT, f"{name}.csv"))
