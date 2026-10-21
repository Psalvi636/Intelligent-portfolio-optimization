import itertools
import json
import os
import time

import pandas as pd

from backtest import load_features, prepare_weekly_inputs, run_backtest


# Output files
OUTPUT_FILE = "results/fuzzy_ga_tuning_results.csv"
SUMMARY_FILE = "results/fuzzy_ga_tuning_summary.csv"
ROBUSTNESS_FILE = "results/fuzzy_ga_robustness_results.csv"
ROBUSTNESS_SUMMARY_FILE = "results/fuzzy_ga_robustness_summary.csv"
FINAL_MODEL_FILE = "results/fuzzy_ga_final_model.json"


# Stage 1: Hyperparameter search
POPULATION_SIZES = [30, 50, 100]
GENERATION_COUNTS = [30, 50]
MUTATION_RATES = [0.05, 0.10]
FUZZY_WEIGHTS = [0.20, 0.30]

# Stage 2: Robustness testing
SEARCH_SEED = 42
RANDOM_SEEDS = [42, 123, 2026]
TOP_N_CONFIGURATIONS = 2

PARAMETER_COLUMNS = [
    "population_size",
    "generations",
    "mutation_rate",
    "fuzzy_weight",
]


def evaluate_configuration(weekly_inputs, parameters, random_seed):
    """Evaluate one configuration using the shared backtester."""

    parameters["population_size"] = int(parameters["population_size"])
    parameters["generations"] = int(parameters["generations"])
    parameters["mutation_rate"] = float(parameters["mutation_rate"])
    parameters["fuzzy_weight"] = float(parameters["fuzzy_weight"])

    results = run_backtest(
        weekly_inputs,
        ga_parameters=parameters,
        random_seed=random_seed,
    )

    metrics = dict(results["strategy_metrics"])

    turnovers = results["turnovers"]
    metrics["average_weekly_turnover"] = (
        sum(turnovers) / len(turnovers) if turnovers else 0.0
    )

    return metrics


def run_search(weekly_inputs):
    """Search all parameter combinations using one fixed seed."""

    configurations = list(
        itertools.product(
            POPULATION_SIZES,
            GENERATION_COUNTS,
            MUTATION_RATES,
            FUZZY_WEIGHTS,
        )
    )

    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        results = existing.to_dict("records")

        completed = {
            tuple(
                (
                    int(row["population_size"]),
                    int(row["generations"]),
                    float(row["mutation_rate"]),
                    float(row["fuzzy_weight"]),
                )
            )
            for row in results
        }

        print(
            f"Existing search results found: "
            f"{len(results)} runs loaded."
        )
    else:
        results = []
        completed = set()

    total_runs = len(configurations)

    print("\nSTAGE 1: HYPERPARAMETER SEARCH")
    print("=" * 65)
    print(f"Configurations: {total_runs}")
    print(f"Random seed: {SEARCH_SEED}\n")

    for run_number, values in enumerate(configurations, start=1):

        parameters = dict(zip(PARAMETER_COLUMNS, values))

        configuration_key = (
            int(parameters["population_size"]),
            int(parameters["generations"]),
            float(parameters["mutation_rate"]),
            float(parameters["fuzzy_weight"]),
        )

        if configuration_key in completed:
            print(
                f"[{run_number:02d}/{total_runs}] "
                f"pop={parameters['population_size']}, "
                f"gen={parameters['generations']}, "
                f"mut={parameters['mutation_rate']:.2f}, "
                f"fuzzy={parameters['fuzzy_weight']:.2f} | "
                f"SKIPPED (already completed)"
            )
            continue

        metrics = evaluate_configuration(
            weekly_inputs,
            parameters,
            SEARCH_SEED,
        )

        row = {
            **parameters,
            "random_seed": SEARCH_SEED,
            **metrics,
        }

        results.append(row)
        completed.add(configuration_key)

        # Save immediately so the run can resume after interruption.
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print(
            f"[{run_number:02d}/{total_runs}] "
            f"pop={parameters['population_size']}, "
            f"gen={parameters['generations']}, "
            f"mut={parameters['mutation_rate']:.2f}, "
            f"fuzzy={parameters['fuzzy_weight']:.2f} | "
            f"Sharpe={metrics['sharpe_ratio']:.3f}, "
            f"Return={metrics['total_return']:.2%}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = results_df[
        PARAMETER_COLUMNS
        + [
            "sharpe_ratio",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "maximum_drawdown",
            "average_weekly_turnover",
        ]
    ].rename(
        columns={
            "sharpe_ratio": "search_sharpe",
            "total_return": "search_total_return",
            "annualized_return": "search_annualized_return",
            "annualized_volatility": "search_annualized_volatility",
            "maximum_drawdown": "search_maximum_drawdown",
            "average_weekly_turnover": "search_average_weekly_turnover",
        }
    )

    summary = summary.sort_values(
        ["search_sharpe", "search_total_return"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print("\nTop configurations from the initial search:")
    print(
        summary.head(TOP_N_CONFIGURATIONS).to_string(
            index=False,
            formatters={
                "mutation_rate": "{:.2f}".format,
                "fuzzy_weight": "{:.2f}".format,
                "search_sharpe": "{:.3f}".format,
                "search_total_return": "{:.2%}".format,
                "search_annualized_return": "{:.2%}".format,
                "search_annualized_volatility": "{:.2%}".format,
                "search_maximum_drawdown": "{:.2%}".format,
                "search_average_weekly_turnover": "{:.2%}".format,
            },
        )
    )

    eligible = summary.dropna(
        subset=["search_sharpe"]
    )

    if len(eligible) < TOP_N_CONFIGURATIONS:
        raise RuntimeError(
            "Fewer than two configurations have valid Sharpe ratios. "
            "Inspect the search results before selecting a final model."
        )

    top_configurations = eligible.head(
        TOP_N_CONFIGURATIONS
    )[PARAMETER_COLUMNS].copy()

    return top_configurations


def run_robustness_check(weekly_inputs, top_configurations):
    """Evaluate the top two configurations using three random seeds."""

    if os.path.exists(ROBUSTNESS_FILE):
        existing = pd.read_csv(ROBUSTNESS_FILE)
        results = existing.to_dict("records")

        completed = {
            (
                int(row["configuration_id"]),
                int(row["random_seed"]),
            )
            for row in results
        }

        print(
            f"Existing robustness results found: "
            f"{len(results)} runs loaded."
        )
    else:
        results = []
        completed = set()

    total_runs = (
        len(top_configurations) * len(RANDOM_SEEDS)
    )
    run_number = 0

    print("\nSTAGE 2: ROBUSTNESS CHECK")
    print("=" * 65)
    print(f"Configurations: {len(top_configurations)}")
    print(f"Random seeds: {RANDOM_SEEDS}")
    print(f"Total robustness runs: {total_runs}\n")

    for configuration_number, (_, row) in enumerate(
        top_configurations.iterrows(),
        start=1,
    ):
        parameters = {
            column: row[column].item()
            if hasattr(row[column], "item")
            else row[column]
            for column in PARAMETER_COLUMNS
        }

        for random_seed in RANDOM_SEEDS:
            run_number += 1

            if (
                configuration_number,
                random_seed,
            ) in completed:

                print(
                    f"[{run_number:02d}/{total_runs}] "
                    f"Config={configuration_number}, "
                    f"seed={random_seed} | "
                    f"SKIPPED (already completed)"
                )
                continue

            metrics = evaluate_configuration(
                weekly_inputs,
                parameters,
                random_seed,
            )

            results.append({
                "configuration_id": configuration_number,
                **parameters,
                "random_seed": random_seed,
                **metrics,
            })

            completed.add(
                (
                    configuration_number,
                    random_seed,
                )
            )

            # Save immediately after every robustness run.
            pd.DataFrame(results).to_csv(
                ROBUSTNESS_FILE,
                index=False,
            )

            print(
                f"[{run_number:02d}/{total_runs}] "
                f"Config={configuration_number}, "
                f"seed={random_seed} | "
                f"Sharpe={metrics['sharpe_ratio']:.3f}, "
                f"Return={metrics['total_return']:.2%}"
            )

    robustness_df = pd.DataFrame(results)

    robustness_df.to_csv(
        ROBUSTNESS_FILE,
        index=False,
    )

    group_columns = [
        "configuration_id",
        *PARAMETER_COLUMNS,
    ]

    summary = (
        robustness_df.groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            mean_sharpe=("sharpe_ratio", "mean"),
            std_sharpe=("sharpe_ratio", "std"),
            min_sharpe=("sharpe_ratio", "min"),
            max_sharpe=("sharpe_ratio", "max"),
            mean_total_return=("total_return", "mean"),
            std_total_return=("total_return", "std"),
            mean_annualized_return=(
                "annualized_return",
                "mean",
            ),
            mean_annualized_volatility=(
                "annualized_volatility",
                "mean",
            ),
            mean_maximum_drawdown=(
                "maximum_drawdown",
                "mean",
            ),
            mean_weekly_turnover=(
                "average_weekly_turnover",
                "mean",
            ),
        )
    )

    summary = summary.sort_values(
        [
            "mean_sharpe",
            "std_sharpe",
            "mean_total_return",
        ],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)

    summary.to_csv(
        ROBUSTNESS_SUMMARY_FILE,
        index=False,
    )

    print("\nRobustness comparison:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "mutation_rate": "{:.2f}".format,
                "fuzzy_weight": "{:.2f}".format,
                "mean_sharpe": "{:.3f}".format,
                "std_sharpe": "{:.3f}".format,
                "min_sharpe": "{:.3f}".format,
                "max_sharpe": "{:.3f}".format,
                "mean_total_return": "{:.2%}".format,
                "std_total_return": "{:.2%}".format,
                "mean_annualized_return": "{:.2%}".format,
                "mean_annualized_volatility": "{:.2%}".format,
                "mean_maximum_drawdown": "{:.2%}".format,
                "mean_weekly_turnover": "{:.2%}".format,
            },
        )
    )

    valid = summary.dropna(
        subset=["mean_sharpe"]
    )

    if valid.empty:
        raise RuntimeError(
            "Neither finalist has a valid mean Sharpe ratio. "
            "No final model was selected."
        )

    selected = valid.iloc[0]

    final_parameters = {
        "max_weight": 0.40,
        "population_size": int(
            selected["population_size"]
        ),
        "generations": int(
            selected["generations"]
        ),
        "mutation_rate": float(
            selected["mutation_rate"]
        ),
        "fuzzy_weight": float(
            selected["fuzzy_weight"]
        ),
        "random_seed": SEARCH_SEED,
    }

    final_model = {
        "model": "Fuzzy-GA",
        "selection_criterion": (
            "Highest mean validation Sharpe ratio across three seeds; "
            "ties broken by lower Sharpe standard deviation, then "
            "higher mean total return."
        ),
        "robustness_seeds": RANDOM_SEEDS,
        "parameters": final_parameters,
        "robustness_metrics": {
            "mean_sharpe": float(
                selected["mean_sharpe"]
            ),
            "std_sharpe": (
                float(selected["std_sharpe"])
                if pd.notna(selected["std_sharpe"])
                else None
            ),
            "mean_total_return": float(
                selected["mean_total_return"]
            ),
            "mean_annualized_return": float(
                selected["mean_annualized_return"]
            ),
            "mean_annualized_volatility": float(
                selected["mean_annualized_volatility"]
            ),
            "mean_maximum_drawdown": float(
                selected["mean_maximum_drawdown"]
            ),
            "mean_weekly_turnover": float(
                selected["mean_weekly_turnover"]
            ),
        },
    }

    with open(
        FINAL_MODEL_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_model,
            file,
            indent=4,
            allow_nan=False,
        )

    print("\nFINAL MODEL SELECTED")
    print("=" * 65)

    for name, value in final_parameters.items():
        print(f"{name:20s}: {value}")

    print(
        f"\nMean validation Sharpe: "
        f"{selected['mean_sharpe']:.3f}"
    )
    print(
        f"Mean total return: "
        f"{selected['mean_total_return']:.2%}"
    )
    print(
        f"Mean maximum drawdown: "
        f"{selected['mean_maximum_drawdown']:.2%}"
    )

    print("\nSaved outputs:")
    print(f"Search results:       {OUTPUT_FILE}")
    print(f"Search summary:       {SUMMARY_FILE}")
    print(f"Robustness results:   {ROBUSTNESS_FILE}")
    print(f"Robustness summary:   {ROBUSTNESS_SUMMARY_FILE}")
    print(f"Final model:          {FINAL_MODEL_FILE}")

    return final_model


def main():
    start_time = time.time()

    os.makedirs(
        "results",
        exist_ok=True,
    )

    print("Preparing validation data...")
    data = load_features()
    weekly_inputs = prepare_weekly_inputs(data)

    print(
        f"Validation weeks: "
        f"{len(weekly_inputs)}"
    )
    print(
        "Validation runs use the shared backtester."
    )

    top_configurations = run_search(
        weekly_inputs
    )

    run_robustness_check(
        weekly_inputs,
        top_configurations,
    )

    print(
        f"\nTotal elapsed time: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )

if __name__ == "__main__":
    main()