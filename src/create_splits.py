import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "splits.json"

features = pd.read_csv(INPUT_FILE, parse_dates=["Date"])
dates = pd.DatetimeIndex(
    sorted(features["Date"].dropna().unique())
)

if dates.empty:
    raise ValueError("No feature dates found.")


periods = [
    ("train", "2016-01-01", "2022-12-31"),
    ("validation", "2023-01-01", "2024-12-31"),
    ("test", "2025-01-01", "2025-12-31"),
]

splits = {}

for name, start, end in periods:
    period_dates = dates[
        (dates >= pd.Timestamp(start))
        & (dates <= pd.Timestamp(end))
    ]

    if period_dates.empty:
        raise ValueError(f"No dates found for {name}.")

    splits[name] = {
        "start": period_dates[0].strftime("%Y-%m-%d"),
        "end": period_dates[-1].strftime("%Y-%m-%d"),
    }

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(splits, f, indent=2)