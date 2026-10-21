import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.environment import PortfolioEnvironment


class WeeklyPortfolioGymEnv(gym.Env):
    """Weekly portfolio environment with drift-aware holdings."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        split="train",
        base_environment=None,
        risk_aversion=0.10,
        turnover_penalty=0.0,
    ):
        super().__init__()

        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split: {split}")

        self.base = base_environment or PortfolioEnvironment()
        self.split = split
        self.risk_aversion = float(risk_aversion)
        self.turnover_penalty = float(turnover_penalty)

        self.n_assets = self.base.n_assets
        self.n_features = len(self.base.feature_columns)
        self.max_weight = self.base.max_weight
        self.market_size = self.n_assets * self.n_features

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.market_size + self.n_assets,),
            dtype=np.float32,
        )

        self.transitions = self._build_transitions()

        if not self.transitions:
            raise ValueError(f"No transitions found for split={split}")

        self.step_index = 0
        self.weights = np.full(
            self.n_assets,
            1.0 / self.n_assets,
            dtype=np.float32,
        )

    def _build_transitions(self):
        """Build weekly holding periods using the common split dates."""

        split_info = self.base.splits[self.split]
        split_start = pd.Timestamp(split_info["start"])
        split_end = pd.Timestamp(split_info["end"])

        dates = pd.DatetimeIndex(
            sorted(self.base.features["Date"].unique())
        )

        split_dates = dates[
            (dates >= split_start) & (dates <= split_end)
        ]

        weekly_groups = (
            pd.DataFrame({"Date": split_dates})
            .assign(Week=split_dates.to_period("W-FRI"))
            .groupby("Week")["Date"]
        )

        transitions = []

        for _, week_dates in weekly_groups:
            holding_start = week_dates.min()
            holding_end = week_dates.max()

            previous_dates = dates[dates < holding_start]

            if len(previous_dates) == 0:
                continue

            decision_date = previous_dates[-1]

            transitions.append(
                (
                    pd.Timestamp(decision_date),
                    pd.Timestamp(holding_start),
                    pd.Timestamp(holding_end),
                )
            )

        return transitions

    def _observation(self):
        date = self.transitions[self.step_index][0]

        data = (
            self.base.features[
                self.base.features["Date"] == date
            ]
            .set_index("Stock")
            .reindex(self.base.stocks)
        )

        market = data[
            self.base.feature_columns
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(market).all():
            market = np.nan_to_num(
                market,
                nan=0.0,
                posinf=10.0,
                neginf=-10.0,
            )

        market[:, 0:3] *= 100.0
        market[:, 3] /= 100.0
        market[:, 4] = (market[:, 4] - 1.0) * 100.0
        market = np.clip(market, -10.0, 10.0)

        return np.concatenate(
            [market.flatten(), self.weights]
        ).astype(np.float32)

    def project_weights(self, action):
        """Project an action onto the constrained portfolio simplex."""

        action = np.asarray(action, dtype=np.float64).reshape(-1)

        if action.size != self.n_assets:
            raise ValueError(
                "Action has the wrong number of assets."
            )

        action = np.nan_to_num(
            action,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        action = np.clip(action, -1.0, 1.0)

        cap = self.max_weight - 1e-7

        low = action.min() - cap
        high = action.max()

        for _ in range(100):
            threshold = (low + high) / 2.0
            candidate = np.clip(
                action - threshold,
                0.0,
                cap,
            )

            if candidate.sum() > 1.0:
                low = threshold
            else:
                high = threshold

        weights = np.clip(
            action - (low + high) / 2.0,
            0.0,
            cap,
        )

        residual = 1.0 - weights.sum()

        for index in np.argsort(-weights):
            if residual <= 1e-12:
                break

            addition = min(
                residual,
                cap - weights[index],
            )
            weights[index] += addition
            residual -= addition

        if abs(weights.sum() - 1.0) > 1e-8:
            raise ValueError(
                "Could not project weights onto the simplex: "
                f"sum={weights.sum():.12f}"
            )

        weights = weights.astype(np.float32)
        self.base.validate_weights(weights)

        return weights

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_index = 0
        self.weights = np.full(
            self.n_assets,
            1.0 / self.n_assets,
            dtype=np.float32,
        )

        return self._observation(), {}

    def step(self, action):
        decision_date, holding_start, holding_end = (
            self.transitions[self.step_index]
        )

        old_weights = self.weights.astype(float).copy()
        new_weights = self.project_weights(action).astype(float)

        returns = (
            self.base.get_weekly_returns(
                holding_start,
                holding_end,
            )
            .reindex(columns=self.base.stocks)
        )

        if returns.empty or returns.isna().any().any():
            raise ValueError(
                f"Missing returns from {holding_start} to {holding_end}"
            )

        daily_returns = returns.to_numpy(dtype=float)

        turnover = float(
            self.base.calculate_turnover(
                new_weights,
                old_weights,
            )
        )

        transaction_cost = float(
            self.base.transaction_cost * turnover
        )

        current_weights = new_weights.copy()
        daily_portfolio_returns = []

        for daily_return in daily_returns:
            portfolio_daily_return = float(
                current_weights @ daily_return
            )

            daily_portfolio_returns.append(
                portfolio_daily_return
            )

            growth_factor = 1.0 + portfolio_daily_return

            if growth_factor <= 0:
                raise ValueError(
                    "Portfolio wealth became non-positive."
                )

            current_weights = (
                current_weights
                * (1.0 + daily_return)
                / growth_factor
            )

        daily_portfolio_returns = np.asarray(
            daily_portfolio_returns,
            dtype=float,
        )

        gross_return = float(
            np.prod(1.0 + daily_portfolio_returns) - 1.0
        )

        net_return = float(
            (1.0 - transaction_cost)
            * (1.0 + gross_return)
            - 1.0
        )

        volatility = (
            float(np.std(daily_portfolio_returns, ddof=1))
            if len(daily_portfolio_returns) > 1
            else 0.0
        )

        reward = 100.0 * (
            net_return
            - self.risk_aversion * volatility
            - self.turnover_penalty * turnover
        )

        ending_weights = current_weights.astype(np.float32)
        self.weights = ending_weights
        self.step_index += 1

        terminated = self.step_index >= len(self.transitions)

        info = {
            "observation_date": decision_date.strftime("%Y-%m-%d"),
            "holding_start": holding_start.strftime("%Y-%m-%d"),
            "holding_end": holding_end.strftime("%Y-%m-%d"),
            "gross_return": gross_return,
            "net_return": net_return,
            "volatility": volatility,
            "turnover": turnover,
            "transaction_cost": transaction_cost,
            "weights": new_weights.copy(),
            "ending_weights": ending_weights.copy(),
        }

        if terminated:
            observation = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )
        else:
            observation = self._observation()

        return observation, float(reward), terminated, False, info