from pathlib import Path

import yfinance as yf

TICKERS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "TCS.NS",
    "HINDUNILVR.NS",
    "SUNPHARMA.NS",
    "LT.NS",
]

START_DATE = "2016-01-01"
END_DATE = "2026-01-01"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "stock_data_raw.csv"

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
data.to_csv(OUTPUT_FILE)