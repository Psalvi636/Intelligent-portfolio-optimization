from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "adjusted_prices.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "features.csv"

RETURN_WINDOW = 20
VOLATILITY_WINDOW = 20
RSI_WINDOW = 14
MA_WINDOW = 20

def calculate_rsi(prices, window=14):
    delta = prices.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window).mean()
    average_loss = losses.rolling(window).mean()

    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))

prices = pd.read_csv(
    INPUT_FILE,
    index_col=0,
    parse_dates=True,
).sort_index()

daily_return = prices.pct_change()
return_20d = prices.pct_change(periods=RETURN_WINDOW)

volatility_20d = daily_return.rolling(
    VOLATILITY_WINDOW
).std()

rsi_14 = prices.apply(
    calculate_rsi,
    window=RSI_WINDOW,
)

moving_average_20d = prices.rolling(
    MA_WINDOW
).mean()

ma_ratio = prices / moving_average_20d

features = []

for stock in prices.columns:
    features.append(
        pd.DataFrame({
            "Date": prices.index,
            "Stock": stock,
            "Daily_Return": daily_return[stock].values,
            "Return_20D": return_20d[stock].values,
            "Volatility_20D": volatility_20d[stock].values,
            "RSI_14": rsi_14[stock].values,
            "MA_Ratio": ma_ratio[stock].values,
        })
    )

features = pd.concat(
    features,
    ignore_index=True,
).dropna().reset_index(drop=True)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

features.to_csv(
    OUTPUT_FILE,
    index=False,
)