import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio.environment import PortfolioEnvironment
from rl.rl_environment import WeeklyPortfolioGymEnv


def metrics(values):
    """Calculate performance metrics from weekly net returns."""

    returns = np.asarray(
        values,
        dtype=float,
    )

    if len(returns) == 0:
        raise ValueError(
            "Cannot calculate metrics for an empty return series."
        )

    if not np.isfinite(returns).all():
        raise ValueError(
            "Return series contains non-finite values."
        )

    if np.any(returns < -1):
        raise ValueError(
            "A return cannot be below -100%."
        )

    wealth = np.concatenate(
        [
            [1.0],
            np.cumprod(1.0 + returns),
        ]
    )

    total_return = wealth[-1] - 1.0

    annualized_return = (
        wealth[-1] ** (52.0 / len(returns)) - 1.0
        if wealth[-1] > 0
        else float("nan")
    )

    std = (
        np.std(
            returns,
            ddof=1,
        )
        if len(returns) > 1
        else 0.0
    )

    annualized_volatility = (
        std * np.sqrt(52.0)
    )

    sharpe = (
        np.mean(returns)
        / std
        * np.sqrt(52.0)
        if std > 0
        else 0.0
    )

    drawdown = (
        wealth
        / np.maximum.accumulate(wealth)
        - 1.0
    )

    return {
        "weeks": int(len(returns)),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "annualized_volatility": float(
            annualized_volatility
        ),
        "sharpe_ratio": float(sharpe),
        "maximum_drawdown": float(
            drawdown.min()
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO portfolio model."
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "test",
        ],
        default="validation",
    )

    parser.add_argument(
        "--model",
        default="ppo_final",
        help="Model filename in models/, with or without .zip",
    )

    parser.add_argument(
        "--tag",
        default=None,
        help="Optional output tag to avoid overwriting earlier results.",
    )

    args = parser.parse_args()

    model_name = args.model.removesuffix(
        ".zip"
    )

    model_path = (
        PROJECT_ROOT
        / "models"
        / f"{model_name}.zip"
    )

    if not model_path.exists():
        alternate_path = PROJECT_ROOT / "models" / model_name

        if alternate_path.exists():
            model_path = alternate_path
        else:
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )
    print(
        f"Loading model: {model_path}"
    )

    base = PortfolioEnvironment()

    env = WeeklyPortfolioGymEnv(
        split=args.split,
        base_environment=base,
        risk_aversion=0.10,
        turnover_penalty=0.0001,
    )

    model = PPO.load(
        str(model_path),
        device="cpu",
    )

    observation, _ = env.reset(
        seed=42
    )

    records = []

    while True:
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        (
            observation,
            _,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        record = {
            "Observation_Date": info[
                "observation_date"
            ],
            "Holding_Start": info[
                "holding_start"
            ],
            "Holding_End": info[
                "holding_end"
            ],
            "PPO_Gross_Return": float(
                info["gross_return"]
            ),
            "PPO_Net_Return": float(
                info["net_return"]
            ),
            "PPO_Turnover": float(
                info["turnover"]
            ),
            "PPO_Transaction_Cost": float(
                info["transaction_cost"]
            ),
            "PPO_Volatility": float(
                info["volatility"]
            ),
        }

        for stock, weight in zip(
            base.stocks,
            info["weights"],
        ):
            record[
                f"Weight_{stock}"
            ] = float(weight)

        for stock, weight in zip(
            base.stocks,
            info["ending_weights"],
        ):
            record[
                f"PPO_EndingWeight_{stock}"
            ] = float(weight)

        records.append(record)

        if terminated or truncated:
            break

    env.close()

    frame = pd.DataFrame(
        records
    )

    results_dir = (
        PROJECT_ROOT / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        f"_{args.tag}"
        if args.tag
        else ""
    )

    csv_path = (
        results_dir
        / f"ppo_{args.split}{suffix}_weekly.csv"
    )

    json_path = (
        results_dir
        / f"ppo_{args.split}{suffix}_metrics.json"
    )

    frame.to_csv(
        csv_path,
        index=False,
    )

    summary = {
        "split": args.split,
        "model": model_name,
        "weeks": int(len(frame)),
        "transaction_cost_rate": (
            base.transaction_cost
        ),
        "max_weight": (
            base.max_weight
        ),
        "turnover_convention": (
            "One-way turnover = 0.5 * "
            "sum(abs(target_weights - "
            "previous_actual_weights)); "
            "first investment from cash has zero turnover."
        ),
        "holding_period_convention": (
            "Weekly target weights with "
            "daily holdings drift and no "
            "rebalancing within the week."
        ),
        "transaction_cost_convention": (
            "Rebalance cost deducted from "
            "capital at the start of each "
            "holding period."
        ),
        "initial_capital_convention": (
            "Initial portfolio starts as "
            "cash; the first model allocation "
            "is the initial investment."
        ),
        "ppo": metrics(
            frame["PPO_Net_Return"]
        ),
        "ppo_average_weekly_turnover": float(
            frame["PPO_Turnover"].mean()
        ),
        "ppo_total_transaction_costs": float(
            frame["PPO_Transaction_Cost"].sum()
        ),
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    result = summary["ppo"]

    print(
        f"\nModel: {model_name}"
    )

    print(
        f"Evaluation split: {args.split}"
    )

    print(
        f"Weeks evaluated: {len(frame)}"
    )

    print(
        "\nPPO performance"
    )

    print(
        "-" * 45
    )

    print(
        f"Total return:             "
        f"{result['total_return']:.2%}"
    )

    print(
        f"Annualized return:        "
        f"{result['annualized_return']:.2%}"
    )

    print(
        f"Annualized volatility:    "
        f"{result['annualized_volatility']:.2%}"
    )

    print(
        f"Sharpe ratio:             "
        f"{result['sharpe_ratio']:.3f}"
    )

    print(
        f"Maximum drawdown:         "
        f"{result['maximum_drawdown']:.2%}"
    )

    print(
        f"\nAverage weekly turnover:   "
        f"{summary['ppo_average_weekly_turnover']:.2%}"
    )

    print(
        f"Total transaction costs:   "
        f"{summary['ppo_total_transaction_costs']:.2%}"
    )

    print(
        "\nTransaction cost: 0.1% multiplied "
        "by one-way portfolio turnover."
    )

    print(
        "\nNote: Evaluation results are "
        "historical estimates and do not "
        "guarantee future performance."
    )

    print(
        f"\nSaved: {csv_path}"
    )

    print(
        f"Saved: {json_path}"
    )


if __name__ == "__main__":
    main()