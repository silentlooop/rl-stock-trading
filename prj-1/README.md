# Inverted Pendulum DQN

A clean, beginner-friendly Stable-Baselines3 implementation of Deep Q-Network (DQN) for Gymnasium's `CartPole-v1`.

## Why DQN for CartPole-v1?

`CartPole-v1` has a discrete action space: the agent chooses between two actions, push left or push right. DQN is built for exactly that setting because it learns a value for each discrete action and picks the best one.

DDPG solves a different problem. It is designed for continuous control, where the agent must output a real-valued action such as a torque. That is why DDPG is commonly used on environments like `Pendulum-v1`, not `CartPole-v1`.

If you switched to `Pendulum-v1`, the action would be a continuous torque value and the policy would need to output a number rather than selecting one of two actions. In that case, DDPG, TD3, or SAC would be a natural fit.

## Project Layout

- `train.py` trains a DQN agent and saves the model as a `.zip` file.
- `evaluate.py` loads the saved model and reports average reward over N episodes.
- `plot_rewards.py` reads SB3 Monitor logs and plots reward per episode.
- `model/` stores trained checkpoints and final models.
- `plots/` stores reward curve images.
- `logs/` stores Monitor and TensorBoard output.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py \
  --total-timesteps 100000 \
  --learning-rate 1e-4 \
  --gamma 0.99 \
  --exploration-fraction 0.1 \
  --buffer-size 50000
```

Useful parameters:

- `--learning-rate`: how aggressively the Q-network updates its weights.
- `--gamma`: how much the agent cares about future reward.
- `--exploration-fraction`: how long epsilon-greedy exploration stays high.
- `--buffer-size`: how many past transitions can be reused from replay.
- `--batch-size`: how many samples are drawn for each gradient update.
- `--learning-starts`: how many steps to collect before learning begins.
- `--target-update-interval`: how often the target network is refreshed.
- `--train-frequency`: how often gradient updates happen.

Training writes episode statistics to `logs/monitor/monitor.csv` and TensorBoard summaries to `logs/tensorboard/`.

## Evaluate

```bash
python evaluate.py --episodes 10
```

To watch the policy act in a window:

```bash
python evaluate.py --episodes 3 --render
```

## Plot Rewards

```bash
python plot_rewards.py
```

This reads the Monitor logs, plots raw episode reward, adds a moving average, and saves the chart to `plots/reward_curve.png`.

## TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

## Saved Model

The trained agent is saved with SB3's built-in `model.save(...)` method and loaded with `DQN.load(...)`.

## Notes

This project is intentionally small and explicit so the DQN mechanics are easy to learn. SB3 handles the replay buffer, target network, epsilon-greedy action selection, and Bellman target updates internally.
