import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_SCRIPT = PROJECT_ROOT / "src" / "rl" / "train_ppo.py"
EVALUATE_SCRIPT = PROJECT_ROOT / "src" / "rl" / "evaluate_ppo.py"

OUTPUT_FILE = PROJECT_ROOT / "results" / "ppo_tuning_results.csv"


TIMESTEPS = 20_000

LEARNING_RATES = [
    1e-4,
    3e-4,
    5e-4,
]

N_STEPS = [
    128,
    256,
]

ENTROPY_COEFFICIENTS = [
    0.0,
    0.01,
]

SEED = 42


def run_command(command):
    """Run a command and stop immediately if it fails."""

    print("\nRunning:")
    print(" ".join(str(x) for x in command))

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}"
        )


def configuration_name(learning_rate, n_steps, ent_coef):
    """Return the shared model name for one configuration."""

    return (
        f"ppo_tune_"
        f"lr{learning_rate:g}_"
        f"steps{n_steps}_"
        f"ent{ent_coef:g}"
    )


def load_existing_result(results_dir, tag):
    """Load an existing evaluation result if available."""

    metrics_file = (
        results_dir
        / f"ppo_validation_{tag}_metrics.json"
    )

    if not metrics_file.exists():
        return None

    with metrics_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def make_result_row(
    learning_rate,
    n_steps,
    ent_coef,
    evaluation,
):
    """Convert PPO evaluation output into one tuning row."""

    ppo = evaluation["ppo"]

    return {
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "ent_coef": ent_coef,
        "timesteps": TIMESTEPS,
        "seed": SEED,
        "total_return": ppo["total_return"],
        "annualized_return": ppo["annualized_return"],
        "annualized_volatility": ppo[
            "annualized_volatility"
        ],
        "sharpe_ratio": ppo["sharpe_ratio"],
        "maximum_drawdown": ppo[
            "maximum_drawdown"
        ],
        "average_weekly_turnover": evaluation[
            "ppo_average_weekly_turnover"
        ],
        "total_transaction_costs": evaluation[
            "ppo_total_transaction_costs"
        ],
    }


def main():
    start_time = time.time()

    results_dir = PROJECT_ROOT / "results"
    models_dir = PROJECT_ROOT / "models"

    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    configurations = list(
        itertools.product(
            LEARNING_RATES,
            N_STEPS,
            ENTROPY_COEFFICIENTS,
        )
    )

    print("PPO hyperparameter tuning")
    print("=" * 50)
    print(f"Configurations: {len(configurations)}")
    print(f"Timesteps per run: {TIMESTEPS}")
    print(f"Seed: {SEED}")

    results = []

    for run_number, (
        learning_rate,
        n_steps,
        ent_coef,
    ) in enumerate(configurations, start=1):

        name = configuration_name(
            learning_rate,
            n_steps,
            ent_coef,
        )

        tag = f"tune_{run_number}"

        model_path = models_dir / f"{name}.zip"

        metrics_file = (
            results_dir
            / f"ppo_validation_{tag}_metrics.json"
        )

        print("\n" + "=" * 70)
        print(
            f"RUN {run_number}/{len(configurations)}"
        )
        print(
            f"learning_rate={learning_rate}, "
            f"n_steps={n_steps}, "
            f"ent_coef={ent_coef}"
        )

        # --------------------------------------------------
        # CASE 1: Evaluation already exists.
        # --------------------------------------------------

        evaluation = load_existing_result(
            results_dir,
            tag,
        )

        if evaluation is not None:
            print(
                "\nExisting evaluation found. "
                "Skipping training and evaluation."
            )

        else:

            # --------------------------------------------------
            # CASE 2: Model exists but evaluation does not.
            # --------------------------------------------------

            if model_path.exists():

                print(
                    "\nExisting model found. "
                    "Skipping training."
                )

            else:

                # --------------------------------------------------
                # CASE 3: Neither model nor evaluation exists.
                # --------------------------------------------------

                train_command = [
                    sys.executable,
                    str(TRAIN_SCRIPT),
                    "--seed",
                    str(SEED),
                    "--timesteps",
                    str(TIMESTEPS),
                    "--name",
                    name,
                    "--learning-rate",
                    str(learning_rate),
                    "--n-steps",
                    str(n_steps),
                    "--batch-size",
                    "64",
                    "--n-epochs",
                    "10",
                    "--gamma",
                    "0.99",
                    "--gae-lambda",
                    "0.95",
                    "--clip-range",
                    "0.2",
                    "--ent-coef",
                    str(ent_coef),
                    "--risk-aversion",
                    "0.10",
                    "--turnover-penalty",
                    "0.0001",
                ]

                run_command(train_command)

            # --------------------------------------------------
            # Evaluate the model.
            # --------------------------------------------------

            if not metrics_file.exists():

                evaluate_command = [
                    sys.executable,
                    str(EVALUATE_SCRIPT),
                    "--split",
                    "validation",
                    "--model",
                    name,
                    "--tag",
                    tag,
                ]

                run_command(evaluate_command)

            evaluation = load_existing_result(
                results_dir,
                tag,
            )

            if evaluation is None:
                raise RuntimeError(
                    f"Evaluation file was not created: "
                    f"{metrics_file}"
                )

        # --------------------------------------------------
        # Store configuration result.
        # --------------------------------------------------

        row = make_result_row(
            learning_rate,
            n_steps,
            ent_coef,
            evaluation,
        )

        results.append(row)

        print("\nResult:")
        print(
            f"Sharpe: "
            f"{row['sharpe_ratio']:.3f}"
        )
        print(
            f"Return: "
            f"{row['total_return']:.2%}"
        )
        print(
            f"Max drawdown: "
            f"{row['maximum_drawdown']:.2%}"
        )
        print(
            f"Turnover: "
            f"{row['average_weekly_turnover']:.2%}"
        )

    # ------------------------------------------------------
    # Create final results table.
    # ------------------------------------------------------

    frame = pd.DataFrame(results)

    frame = frame.sort_values(
        ["sharpe_ratio", "total_return"],
        ascending=False,
    )

    frame.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("PPO TUNING COMPLETE")
    print("=" * 70)

    print(f"Results saved to: {OUTPUT_FILE}")

    print("\nAll configurations ranked by validation Sharpe:")
    print(
        frame.to_string(
            index=False,
            formatters={
                "learning_rate": "{:.0e}".format,
                "ent_coef": "{:.3f}".format,
                "total_return": "{:.2%}".format,
                "annualized_return": "{:.2%}".format,
                "annualized_volatility": "{:.2%}".format,
                "sharpe_ratio": "{:.3f}".format,
                "maximum_drawdown": "{:.2%}".format,
                "average_weekly_turnover": "{:.2%}".format,
                "total_transaction_costs": "{:.2%}".format,
            },
        )
    )

    print("\n" + "=" * 70)
    print("TOP 10 PPO CONFIGURATIONS")
    print("=" * 70)

    top_10 = frame.head(10)

    print(
        top_10.to_string(
            index=False,
            formatters={
                "learning_rate": "{:.0e}".format,
                "ent_coef": "{:.3f}".format,
                "total_return": "{:.2%}".format,
                "annualized_return": "{:.2%}".format,
                "annualized_volatility": "{:.2%}".format,
                "sharpe_ratio": "{:.3f}".format,
                "maximum_drawdown": "{:.2%}".format,
                "average_weekly_turnover": "{:.2%}".format,
                "total_transaction_costs": "{:.2%}".format,
            },
        )
    )

    print(
        f"\nElapsed time: "
        f"{(time.time() - start_time) / 60:.2f} minutes"
    )


if __name__ == "__main__":
    main()