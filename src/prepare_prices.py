import os
import pandas as pd

# File paths
input_file = "data/raw/stock_data_raw.csv"
output_file = "data/processed/adjusted_prices.csv"

# Create processed data directory
os.makedirs("data/processed", exist_ok=True)

# Load raw Yahoo Finance data
data = pd.read_csv(
    input_file,
    header=[0, 1],
    index_col=0
)

# Convert index to datetime
data.index = pd.to_datetime(data.index)

# Extract only Adjusted Close prices
prices = data["Adj Close"].copy()

# Remove the .NS suffix from ticker names
prices.columns = [
    ticker.replace(".NS", "")
    for ticker in prices.columns
]

# Sort columns into our desired order
tickers = [
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "HINDUNILVR",
    "SUNPHARMA",
    "LT"
]

prices = prices[tickers]

# Save clean adjusted-price dataset
prices.to_csv(output_file)