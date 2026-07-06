"""Evaluate a trained CartPole DQN model.

This script loads the SB3 .zip file saved by train.py and runs a fixed number of
episodes. The average reward is reported with SB3's evaluate_policy helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

DEFAULT_MODEL_PATH = Path("model/cartpole_dqn.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN CartPole model")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the saved SB3 model.")
    parser.add_argument("--episodes", type=int, default=10, help="How many evaluation episodes to run.")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment while evaluating. Keep this off for faster headless runs.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the greedy action from the learned policy during evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the evaluation environment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}. Train first or pass --model-path.")

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    env = Monitor(env)
    env.reset(seed=args.seed)

    # evaluate_policy handles the episode loop and returns the mean/std reward.
    model = DQN.load(str(args.model_path), env=env, device="auto")
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
    )

    print(f"Model: {args.model_path}")
    print(f"Episodes: {args.episodes}")
    print(f"Average reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()


if __name__ == "__main__":
    main()
