"""Evaluate the final Fuzzy-GA model on the validation or test split.

Usage (from the project root):
    python src/fuzzy_ga/evaluate.py --split test
    python src/fuzzy_ga/evaluate.py --split validation
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest import (
    load_features,
    load_split,
    prepare_weekly_inputs,
    run_backtest,
)


GA_PARAMETERS = {
    "max_weight": 0.40,
    "population_size": 100,
    "generations": 30,
    "mutation_rate": 0.05,
    "fuzzy_weight": 0.20,
    "random_seed": 42,
}

RESULTS_DIR = Path("results")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the final Fuzzy-GA model."
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
    )
    args = parser.parse_args()

    start, end = load_split(args.split)
    weekly_inputs = prepare_weekly_inputs(load_features(), args.split)
    results = run_backtest(weekly_inputs, GA_PARAMETERS)

    metrics = {
        "model": "Fuzzy-GA",
        "split": args.split,
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "weeks_evaluated": len(results["strategy_returns"]),
        "parameters": GA_PARAMETERS,
        **results["strategy_metrics"],
        "average_weekly_turnover": results["average_weekly_turnover"],
        "total_transaction_costs": results["total_transaction_costs"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    weekly_path = RESULTS_DIR / f"fuzzy_ga_{args.split}_final_with_weights.csv"
    metrics_path = (
        RESULTS_DIR / f"fuzzy_ga_{args.split}_final_with_weights_metrics.json"
    )

    pd.DataFrame(results["records"]).to_csv(weekly_path, index=False)

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"Fuzzy-GA {args.split} ({start.date()} to {end.date()}), "
          f"{metrics['weeks_evaluated']} weeks")
    print(f"Total return:            {metrics['total_return']:.2%}")
    print(f"Annualized return:       {metrics['annualized_return']:.2%}")
    print(f"Annualized volatility:   {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe ratio:            {metrics['sharpe_ratio']:.3f}")
    print(f"Maximum drawdown:        {metrics['maximum_drawdown']:.2%}")
    print(f"Avg weekly turnover:     {metrics['average_weekly_turnover']:.2%}")
    print(f"Total transaction costs: {metrics['total_transaction_costs']:.2%}")
    print(f"Saved: {weekly_path}, {metrics_path}")


if __name__ == "__main__":
    main()