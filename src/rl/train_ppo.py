import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rl.rl_environment import WeeklyPortfolioGymEnv


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO portfolio models with configurable hyperparameters."
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--name",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
    )

    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--risk-aversion",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--turnover-penalty",
        type=float,
        default=0.0001,
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error("--timesteps must be positive.")

    if args.n_steps <= 0:
        parser.error("--n-steps must be positive.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.n_epochs <= 0:
        parser.error("--n-epochs must be positive.")

    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.name or (
        f"ppo_portfolio_seed_{args.seed}_{args.timesteps}"
    )

    model_path = model_dir / model_name

    if model_path.with_suffix(".zip").exists():
        raise FileExistsError(
            f"Model already exists: {model_path}.zip"
        )

    env = WeeklyPortfolioGymEnv(
        split="train",
        risk_aversion=args.risk_aversion,
        turnover_penalty=args.turnover_penalty,
    )

    try:
        print("Checking environment...")
        check_env(env, warn=True)

        print("\nPPO configuration")
        print("-" * 40)
        print(f"Training transitions: {len(env.transitions)}")
        print(f"Seed: {args.seed}")
        print(f"Timesteps: {args.timesteps}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"n_steps: {args.n_steps}")
        print(f"Batch size: {args.batch_size}")
        print(f"n_epochs: {args.n_epochs}")
        print(f"Gamma: {args.gamma}")
        print(f"GAE lambda: {args.gae_lambda}")
        print(f"Clip range: {args.clip_range}")
        print(f"Entropy coefficient: {args.ent_coef}")
        print(f"Risk aversion: {args.risk_aversion}")
        print(f"Turnover penalty: {args.turnover_penalty}")
        print(f"Model name: {model_name}")

        model = PPO(
            "MlpPolicy",
            Monitor(env),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device="cpu",
            verbose=0,
        )

        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=False,
        )

        model.save(str(model_path))

        print("\nTraining completed.")
        print(f"Actual timesteps: {model.num_timesteps}")
        print(f"Saved model: {model_path}.zip")

    finally:
        env.close()


if __name__ == "__main__":
    main()