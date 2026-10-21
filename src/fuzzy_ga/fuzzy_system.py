
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzySuitabilitySystem:
    """Calculate stock suitability from return, volatility, and RSI."""

    def __init__(self):
        self.returns = ctrl.Antecedent(
            np.arange(-0.20, 0.201, 0.001), "returns"
        )
        self.volatility = ctrl.Antecedent(
            np.arange(0.0, 0.101, 0.001), "volatility"
        )
        self.rsi = ctrl.Antecedent(np.arange(0, 101, 1), "rsi")
        self.suitability = ctrl.Consequent(
            np.arange(0, 101, 1), "suitability"
        )

        self._define_membership_functions()
        self.rules = self._define_rules()

        system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(system)

    def _define_membership_functions(self):
        self.returns["low"] = fuzz.trapmf(
            self.returns.universe, [-0.20, -0.20, -0.05, 0.0]
        )
        self.returns["medium"] = fuzz.trimf(
            self.returns.universe, [-0.05, 0.0, 0.05]
        )
        self.returns["high"] = fuzz.trapmf(
            self.returns.universe, [0.0, 0.05, 0.20, 0.20]
        )

        self.volatility["low"] = fuzz.trapmf(
            self.volatility.universe, [0.0, 0.0, 0.01, 0.025]
        )
        self.volatility["medium"] = fuzz.trimf(
            self.volatility.universe, [0.01, 0.03, 0.05]
        )
        self.volatility["high"] = fuzz.trapmf(
            self.volatility.universe, [0.03, 0.05, 0.10, 0.10]
        )

        self.rsi["low"] = fuzz.trapmf(
            self.rsi.universe, [0, 0, 20, 40]
        )
        self.rsi["medium"] = fuzz.trimf(
            self.rsi.universe, [30, 50, 70]
        )
        self.rsi["high"] = fuzz.trapmf(
            self.rsi.universe, [60, 80, 100, 100]
        )

        self.suitability["low"] = fuzz.trimf(
            self.suitability.universe, [0, 0, 50]
        )
        self.suitability["medium"] = fuzz.trimf(
            self.suitability.universe, [25, 50, 75]
        )
        self.suitability["high"] = fuzz.trimf(
            self.suitability.universe, [50, 100, 100]
        )

    def _define_rules(self):
        return [
            ctrl.Rule(
                self.returns["high"] & self.volatility["low"],
                self.suitability["high"],
            ),
            ctrl.Rule(
                self.returns["high"] & self.volatility["medium"],
                self.suitability["high"],
            ),
            ctrl.Rule(
                self.returns["medium"] & self.volatility["low"],
                self.suitability["high"],
            ),
            ctrl.Rule(
                self.returns["medium"] & self.volatility["medium"],
                self.suitability["medium"],
            ),
            ctrl.Rule(
                self.returns["low"], self.suitability["low"]
            ),
            ctrl.Rule(
                self.volatility["high"], self.suitability["low"]
            ),
            ctrl.Rule(
                self.rsi["high"] & self.returns["high"],
                self.suitability["high"],
            ),
            ctrl.Rule(
                self.rsi["low"] & self.returns["low"],
                self.suitability["low"],
            ),
            ctrl.Rule(
                self.rsi["medium"] & self.returns["medium"],
                self.suitability["medium"],
            ),
        ]

    def evaluate(self, return_20d, volatility_20d, rsi_14):
        """Return one suitability score between 0 and 100."""
        values = {
            "returns": float(np.clip(return_20d, -0.20, 0.20)),
            "volatility": float(np.clip(volatility_20d, 0.0, 0.10)),
            "rsi": float(np.clip(rsi_14, 0, 100)),
        }

        self.simulation.reset()

        for name, value in values.items():
            self.simulation.input[name] = value

        self.simulation.compute()
        return float(self.simulation.output["suitability"])

    def evaluate_many(self, returns, volatilities, rsi_values):
        """Calculate scores for multiple stocks using the same system."""
        returns = np.asarray(returns, dtype=float)
        volatilities = np.asarray(volatilities, dtype=float)
        rsi_values = np.asarray(rsi_values, dtype=float)

        if not (
            returns.ndim == volatilities.ndim == rsi_values.ndim == 1
            and returns.shape == volatilities.shape == rsi_values.shape
        ):
            raise ValueError("Fuzzy input arrays must have matching shapes.")

        if not (
            np.all(np.isfinite(returns))
            and np.all(np.isfinite(volatilities))
            and np.all(np.isfinite(rsi_values))
        ):
            raise ValueError("Fuzzy inputs must contain finite values.")

        return np.array([
            self.evaluate(ret, vol, rsi)
            for ret, vol, rsi in zip(
                returns, volatilities, rsi_values
            )
        ])
