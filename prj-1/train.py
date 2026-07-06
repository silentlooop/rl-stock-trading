"""Train a DQN agent on Gymnasium's CartPole-v1.

Why DQN here?
- CartPole-v1 has a discrete action space: action 0 means push left, action 1 means push right.
- DQN is designed to choose among a finite set of actions by learning Q(s, a) values.
- DDPG is usually used when the action space is continuous, where the agent must output a real-valued action.

If you wanted DDPG instead, you'd typically switch to a continuous-control environment such as
Pendulum-v1, where the action is a torque value rather than a small discrete choice.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

DEFAULT_MODEL_PATH = Path("model/cartpole_dqn.zip")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_TENSORBOARD_DIR = DEFAULT_LOG_DIR / "tensorboard"
DEFAULT_MONITOR_DIR = DEFAULT_LOG_DIR / "monitor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN on CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total environment steps to train for.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer step size for the Q-network.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards.")
    parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Portion of training where epsilon is annealed from 1.0 to exploration_final_eps.",
    )
    parser.add_argument(
        "--exploration-initial-eps",
        type=float,
        default=1.0,
        help="Initial epsilon for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--exploration-final-eps",
        type=float,
        default=0.05,
        help="Final epsilon after exploration_fraction of training.",
    )
    parser.add_argument("--buffer-size", type=int, default=50_000, help="Replay buffer capacity.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size sampled from replay buffer.")
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1_000,
        help="Number of steps to collect before gradient updates begin.",
    )
    parser.add_argument(
        "--train-frequency",
        type=int,
        default=4,
        help="How often to train: perform one gradient update every N environment steps.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="How many gradient updates to perform each training update.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1_000,
        help="How frequently to copy weights into the target network.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Soft update factor for the target network. 1.0 means hard updates.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Where to save the trained model.")
    parser.add_argument("--tensorboard-log", type=Path, default=DEFAULT_TENSORBOARD_DIR, help="TensorBoard log directory.")
    parser.add_argument("--monitor-dir", type=Path, default=DEFAULT_MONITOR_DIR, help="Directory for episode monitor logs.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Save an intermediate checkpoint every N environment steps.",
    )
    return parser.parse_args()


def make_env(monitor_dir: Path, seed: int) -> gym.Env:
    env = gym.make("CartPole-v1")
    env = Monitor(env, filename=str(monitor_dir / "monitor.csv"))
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def main() -> None:
    args = parse_args()

    set_random_seed(args.seed)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)
    args.monitor_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.monitor_dir, args.seed)

    # DQN learns a Q-value for each discrete action and picks the best-valued action.
    # SB3 handles the replay buffer, target network, and Bellman target updates internally.
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_frequency,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        tau=args.tau,
        verbose=1,
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device="auto",
    )

    checkpoint_dir = args.model_path.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq, 1),
        save_path=str(checkpoint_dir),
        name_prefix="cartpole_dqn",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        tb_log_name="DQN_CartPole_v1",
        progress_bar=True,
    )

    model.save(str(args.model_path))
    print(f"Saved model to {args.model_path}")
    print(f"Monitor logs saved under {args.monitor_dir}")
    print(f"TensorBoard logs saved under {args.tensorboard_log}")

    env.close()


if __name__ == "__main__":
    main()
