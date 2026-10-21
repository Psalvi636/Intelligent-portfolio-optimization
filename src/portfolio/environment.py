import json
import numpy as np
import pandas as pd

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

TRANSACTION_COST = 0.001
MAX_WEIGHT = 0.40


class PortfolioEnvironment:
    """Common weekly portfolio environment for Fuzzy-GA and RL."""

    def __init__(
        self,
        features_file=FEATURES_FILE,
        splits_file=SPLITS_FILE,
        transaction_cost=TRANSACTION_COST,
        max_weight=MAX_WEIGHT,
    ):
        self.features = pd.read_csv(
            features_file,
            parse_dates=["Date"],
        )

        with open(splits_file, "r") as file:
            self.splits = json.load(file)

        self.transaction_cost = transaction_cost
        self.max_weight = max_weight

        self.stocks = STOCKS
        self.n_assets = len(self.stocks)

        self.feature_columns = [
            "Daily_Return",
            "Return_20D",
            "Volatility_20D",
            "RSI_14",
            "MA_Ratio",
        ]

        self._prepare_weekly_dates()

    def _prepare_weekly_dates(self):
        """Identify weekly decision dates using the last available
        trading day of each week."""

        dates = pd.DatetimeIndex(
            sorted(self.features["Date"].unique())
        )

        dates_df = pd.DataFrame({"Date": dates})

        dates_df["Week"] = dates_df["Date"].dt.to_period("W-FRI")

        self.weekly_dates = (
            dates_df
            .groupby("Week")["Date"]
            .max()
            .reset_index(drop=True)
        )

    def get_state(self, date, portfolio_weights):
        """Return the market state and current portfolio weights."""

        date = pd.Timestamp(date)

        data = (
            self.features[
                self.features["Date"] == date
            ]
            .set_index("Stock")
            .loc[self.stocks]
        )

        market_state = data[self.feature_columns].values.flatten()

        state = np.concatenate([
            market_state,
            np.asarray(portfolio_weights, dtype=np.float32),
        ])

        return state.astype(np.float32)

    
    def calculate_turnover(self, new_weights, old_weights):
        """Calculate one-way portfolio turnover.

        For a fully invested portfolio, half the L1 distance
        represents the fraction of portfolio value traded.
        """
        return 0.5 * np.sum(
            np.abs(
                np.asarray(new_weights, dtype=float)
                - np.asarray(old_weights, dtype=float)
            )
        )


    def calculate_transaction_cost(
        self,
        new_weights,
        old_weights,
    ):
        """Calculate transaction cost."""

        turnover = self.calculate_turnover(
            new_weights,
            old_weights,
        )

        return self.transaction_cost * turnover

    def validate_weights(self, weights):
        """Validate portfolio allocation constraints."""

        weights = np.asarray(weights, dtype=float)

        if len(weights) != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} weights."
            )

        if np.any(weights < 0):
            raise ValueError(
                "Portfolio weights cannot be negative."
            )

        if np.any(weights > self.max_weight):
            raise ValueError(
                "Portfolio weight exceeds maximum limit."
            )

        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(
                "Portfolio weights must sum to 1."
            )

        return True
    
    def get_next_trading_date(self, date):
        """Return the first trading date after the given date."""

        dates = pd.DatetimeIndex(
            sorted(self.features["Date"].unique())
        )

        future_dates = dates[dates > pd.Timestamp(date)]

        if len(future_dates) == 0:
            return None

        return future_dates[0]
    
    def get_weekly_returns(self, start_date, end_date):
        """Return daily stock returns for a holding period."""

        data = self.features[
            (self.features["Date"] >= pd.Timestamp(start_date))
            & (self.features["Date"] <= pd.Timestamp(end_date))
        ].copy()

        returns = (
            data
            .pivot(
                index="Date",
                columns="Stock",
                values="Daily_Return",
            )
            .reindex(columns=self.stocks)
        )

        return returns

    def calculate_portfolio_return(
        self,
        weights,
        start_date,
        end_date,
    ):
        """Calculate compounded portfolio return."""

        returns = self.get_weekly_returns(
            start_date,
            end_date,
        )

        daily_portfolio_returns = returns @ np.asarray(weights)

        weekly_return = (
            (1 + daily_portfolio_returns).prod() - 1
        )

        return float(weekly_return)

    def calculate_portfolio_volatility(
        self,
        weights,
        returns,
    ):
        """Calculate portfolio volatility from a return matrix."""

        covariance = returns.cov().values

        weights = np.asarray(weights)

        variance = weights.T @ covariance @ weights

        return float(np.sqrt(max(variance, 0)))

    def calculate_objective(
        self,
        weights,
        old_weights,
        observation_date,
        next_week_end,
        risk_aversion,
    ):
        """Calculate portfolio performance for the week
        following the observation date."""

        observation_date = pd.Timestamp(observation_date)
        next_week_end = pd.Timestamp(next_week_end)

        returns = self.get_weekly_returns(
            self.get_next_trading_date(observation_date),
            next_week_end,
        )

        portfolio_return = self.calculate_portfolio_return(
            weights,
            self.get_next_trading_date(observation_date),
            next_week_end,
        )

        volatility = self.calculate_portfolio_volatility(
            weights,
            returns,
        )

        turnover = self.calculate_turnover(
            weights,
            old_weights,
        )

        transaction_cost = (
            self.transaction_cost * turnover
        )

        objective = (
            portfolio_return
            - risk_aversion * volatility
            - transaction_cost
        )

        return {
            "return": portfolio_return,
            "volatility": volatility,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "objective": objective,
        }