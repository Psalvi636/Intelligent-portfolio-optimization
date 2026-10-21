"""Shared Fuzzy-GA backtesting engine.

Weekly timeline: the portfolio is decided on the last trading day before
each holding week, using only data up to that day, then held for the week
with weights drifting daily. The transaction cost is charged at the start
of the week on one-way turnover.
"""

import json

import numpy as np
import pandas as pd

from fuzzy_system import FuzzySuitabilitySystem
from genetic_algorithm import GeneticAlgorithm


FEATURES_FILE = "data/processed/features.csv"
SPLITS_FILE = "data/splits.json"

STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "HINDUNILVR",
    "SUNPHARMA",
    "LT",
]

LOOKBACK_DAYS = 60
TRADING_DAYS_PER_WEEK = 5
TRANSACTION_COST = 0.001

REQUIRED_COLUMNS = {
    "Date",
    "Stock",
    "Daily_Return",
    "Return_20D",
    "Volatility_20D",
    "RSI_14",
}


def load_features(path=FEATURES_FILE):
    """Load and validate the engineered feature dataset."""
    data = pd.read_csv(path, parse_dates=["Date"])

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.sort_values(["Date", "Stock"]).reset_index(drop=True)

    missing_stocks = set(STOCKS) - set(data["Stock"].unique())
    if missing_stocks:
        raise ValueError(f"Missing stocks: {sorted(missing_stocks)}")

    return data


def load_split(name, path=SPLITS_FILE):
    """Return (start, end) timestamps of a split from splits.json."""
    with open(path, "r", encoding="utf-8") as file:
        split = json.load(file)[name]

    return pd.Timestamp(split["start"]), pd.Timestamp(split["end"])


def pivot_daily_returns(data):
    """Return a Date x Stock matrix of daily returns."""
    return (
        data.pivot(
            index="Date",
            columns="Stock",
            values="Daily_Return",
        )
        .reindex(columns=STOCKS)
        .sort_index()
    )


def prepare_inputs(data, decision_date, fuzzy_system=None, daily_returns=None):
    """Prepare fuzzy scores, expected returns, and covariance from past data."""
    decision_date = pd.Timestamp(decision_date)

    available = data.loc[data["Date"] <= decision_date, "Date"]
    if available.empty:
        raise ValueError("No data is available by the decision date.")

    observation_date = available.max()

    observation = (
        data.loc[data["Date"] == observation_date]
        .set_index("Stock")
        .reindex(STOCKS)
    )

    features = observation[["Return_20D", "Volatility_20D", "RSI_14"]]

    if features.isna().any().any():
        raise ValueError(
            f"Missing fuzzy features at {observation_date.date()}."
        )

    if fuzzy_system is None:
        fuzzy_system = FuzzySuitabilitySystem()

    fuzzy_scores = fuzzy_system.evaluate_many(
        features["Return_20D"].to_numpy(),
        features["Volatility_20D"].to_numpy(),
        features["RSI_14"].to_numpy(),
    )

    if daily_returns is None:
        daily_returns = pivot_daily_returns(data)

    history = daily_returns.loc[
        daily_returns.index <= observation_date, STOCKS
    ].tail(LOOKBACK_DAYS)

    if len(history) < LOOKBACK_DAYS:
        raise ValueError(
            f"Need {LOOKBACK_DAYS} historical days before "
            f"{observation_date.date()}; found {len(history)}."
        )

    if history.isna().any().any():
        raise ValueError(
            f"Missing historical returns at {observation_date.date()}."
        )

    expected_returns = history.mean().to_numpy() * TRADING_DAYS_PER_WEEK
    covariance_matrix = history.cov().to_numpy() * TRADING_DAYS_PER_WEEK

    return (
        observation_date,
        fuzzy_scores,
        expected_returns,
        covariance_matrix,
    )


def prepare_weekly_inputs(data=None, split="validation"):
    """Prepare model inputs and realized returns for each week of a split.

    Returns one dict per week with the decision date, holding dates,
    fuzzy scores, expected returns, covariance matrix and the realized
    daily returns of the holding week.
    """
    if data is None:
        data = load_features()

    start, end = load_split(split)
    daily_returns = pivot_daily_returns(data)

    if daily_returns[STOCKS].isna().any().any():
        raise ValueError("Daily return data contains missing stock values.")

    available_dates = pd.DatetimeIndex(daily_returns.index)
    split_dates = available_dates[
        (available_dates >= start) & (available_dates <= end)
    ]

    if split_dates.empty:
        raise ValueError(f"No dates found in the {split} period.")

    date_frame = pd.DataFrame({"Date": split_dates})
    date_frame["Week"] = date_frame["Date"].dt.to_period("W-FRI")

    fuzzy_system = FuzzySuitabilitySystem()
    weekly_inputs = []

    for _, group in date_frame.groupby("Week", sort=True):
        holding_dates = group["Date"].tolist()
        holding_start = pd.Timestamp(min(holding_dates))
        holding_end = pd.Timestamp(max(holding_dates))

        prior_dates = available_dates[available_dates < holding_start]
        if prior_dates.empty:
            continue

        (
            observation_date,
            fuzzy_scores,
            expected_returns,
            covariance_matrix,
        ) = prepare_inputs(
            data,
            pd.Timestamp(prior_dates[-1]),
            fuzzy_system=fuzzy_system,
            daily_returns=daily_returns,
        )

        realized = daily_returns.loc[
            (daily_returns.index >= holding_start)
            & (daily_returns.index <= holding_end),
            STOCKS,
        ]

        if realized.empty or realized.isna().any().any():
            raise ValueError(
                f"Missing realized returns for week ending "
                f"{holding_end.date()}."
            )

        weekly_inputs.append({
            "decision_date": observation_date,
            "holding_start": holding_start,
            "holding_end": holding_end,
            "fuzzy_scores": fuzzy_scores,
            "expected_returns": expected_returns,
            "covariance_matrix": covariance_matrix,
            "realized_returns": realized.to_numpy(),
        })

    if not weekly_inputs:
        raise ValueError(f"No {split} weeks could be prepared.")

    return weekly_inputs


def performance_metrics(weekly_returns):
    """Calculate performance metrics from weekly portfolio returns."""
    returns = np.asarray(weekly_returns, dtype=float)

    if returns.size == 0:
        raise ValueError("Cannot evaluate an empty return series.")
    if not np.all(np.isfinite(returns)):
        raise ValueError("Returns contain non-finite values.")
    if np.any(returns < -1):
        raise ValueError("A return cannot be below -100%.")

    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    total_return = equity[-1] - 1.0

    annualized_return = (
        equity[-1] ** (52 / len(returns)) - 1.0
        if equity[-1] > 0
        else np.nan
    )

    weekly_std = (
        np.std(returns, ddof=1) if len(returns) > 1 else np.nan
    )
    annualized_volatility = (
        weekly_std * np.sqrt(52)
        if np.isfinite(weekly_std)
        else np.nan
    )
    sharpe_ratio = (
        np.mean(returns) / weekly_std * np.sqrt(52)
        if np.isfinite(weekly_std) and weekly_std > 0
        else np.nan
    )

    running_peak = np.maximum.accumulate(equity)
    maximum_drawdown = np.min(equity / running_peak - 1.0)

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(annualized_volatility),
        "sharpe_ratio": float(sharpe_ratio),
        "maximum_drawdown": float(maximum_drawdown),
    }


def run_backtest(weekly_inputs, ga_parameters=None, random_seed=42):
    """Run the Fuzzy-GA weekly rebalancing backtest.

    Parameters not given in ga_parameters use the GeneticAlgorithm
    defaults. A random_seed inside ga_parameters overrides random_seed.

    Returns weekly returns, turnovers, transaction costs and portfolio
    values, summary metrics, and one record per week with dates, target
    weights (before drift) and ending weights (after drift).
    """
    parameters = {"random_seed": random_seed}
    parameters.update(ga_parameters or {})
    parameters["n_assets"] = len(STOCKS)

    ga = GeneticAlgorithm(**parameters)

    current_weights = None
    strategy_returns = []
    turnovers = []
    transaction_costs = []
    portfolio_values = [1.0]
    records = []

    for week in weekly_inputs:
        result = ga.optimize(
            fuzzy_scores=week["fuzzy_scores"],
            expected_returns=week["expected_returns"],
            covariance_matrix=week["covariance_matrix"],
        )

        target_weights = result["weights"]

        if not ga.validate_weights(target_weights):
            raise RuntimeError(
                f"Invalid portfolio weights at "
                f"{week['decision_date'].date()}."
            )

        if current_weights is None:
            turnover = 0.0
        else:
            turnover = 0.5 * float(
                np.abs(target_weights - current_weights).sum()
            )

        cost = TRANSACTION_COST * turnover

        current_weights = target_weights.copy()
        weekly_growth = 1.0

        for daily_returns in week["realized_returns"]:
            daily_return = np.dot(current_weights, daily_returns)
            weekly_growth *= 1.0 + daily_return
            current_weights *= 1.0 + daily_returns
            current_weights /= current_weights.sum()

        gross_return = weekly_growth - 1.0
        net_return = (1.0 + gross_return) * (1.0 - cost) - 1.0

        portfolio_values.append(
            portfolio_values[-1] * (1.0 + net_return)
        )
        strategy_returns.append(float(net_return))
        turnovers.append(float(turnover))
        transaction_costs.append(float(cost))

        record = {
            "decision_date": week["decision_date"].date().isoformat(),
            "holding_start": week["holding_start"].date().isoformat(),
            "holding_end": week["holding_end"].date().isoformat(),
            "weekly_return": float(net_return),
            "turnover": float(turnover),
            "transaction_cost": float(cost),
            "portfolio_value": float(portfolio_values[-1]),
        }

        for stock, weight in zip(STOCKS, target_weights):
            record[f"target_weight_{stock}"] = float(weight)

        for stock, weight in zip(STOCKS, current_weights):
            record[f"ending_weight_{stock}"] = float(weight)

        records.append(record)

    return {
        "strategy_returns": strategy_returns,
        "turnovers": turnovers,
        "transaction_costs": transaction_costs,
        "portfolio_values": portfolio_values,
        "records": records,
        "strategy_metrics": performance_metrics(strategy_returns),
        "average_weekly_turnover": float(np.mean(turnovers)),
        "total_transaction_costs": float(np.sum(transaction_costs)),
    }