# Technical Explanation

## Why DQN Works Here

`CartPole-v1` is a discrete-control problem. At every step the agent has only two actions: push left or push right. That makes it a good fit for DQN because DQN estimates a Q-value for each possible action:

$$
Q(s, a) = \text{expected discounted return from taking action } a \text{ in state } s
$$

The policy is then simple: choose the action with the largest Q-value.

DDPG exists for the opposite case: continuous action spaces. Instead of choosing from a small set of actions, the actor outputs a real-valued action directly. On `Pendulum-v1`, for example, the action is a torque value, so a continuous-control method like DDPG, TD3, or SAC is appropriate.

## What the Bellman Update Looks Like

DQN learns from the Bellman optimality equation. Conceptually, the target for one transition is:

$$
 y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
$$

Where:
- $r$ is the immediate reward
- $\gamma$ is the discount factor
- $s'$ is the next state
- $Q_{\text{target}}$ is the target network

The online network is trained to make $Q(s, a)$ match that target. Intuitively, DQN says: "my current estimate for this action should move toward the reward I got plus the best future value I can expect from the next state."

## Why Stable-Baselines3 Uses a Target Network and Replay Buffer

Even though this project does not implement them from scratch, SB3's DQN uses both internally because they make training much more stable.

The replay buffer stores past transitions and samples random mini-batches from them. That breaks up the strong correlation between consecutive frames from the environment, which makes gradient descent behave better.

The target network is a slowly updated copy of the online Q-network. Without it, the target values would move every time the network changes, which makes training chase a moving target and can cause instability or divergence.

Together, replay and target networks reduce oscillation and improve sample efficiency.

## Hyperparameters That Matter Most

### `learning_rate`
Higher values learn faster but can become unstable. If rewards fluctuate wildly or training diverges, this is one of the first values to reduce.

### `gamma`
This controls how much the agent values future reward. For CartPole, values near `0.99` are common because keeping the pole upright depends on planning ahead.

### `buffer_size`
A larger replay buffer gives more diverse past experience, which often improves stability. Too small a buffer can overfit recent behavior.

### `batch_size`
Larger batches produce smoother gradient estimates but use more memory and can reduce update frequency if training is slow.

### `exploration_fraction`, `exploration_initial_eps`, `exploration_final_eps`
These control epsilon-greedy exploration. High exploration early on helps the agent discover useful trajectories; lowering epsilon too quickly can trap the policy in a weak strategy.

### `learning_starts`
DQN should not update immediately from a tiny buffer. A warm-up period lets the replay buffer collect enough varied transitions first.

### `train_freq` and `gradient_steps`
These determine how often the network updates relative to environment interaction. More updates can improve learning speed, but too many can overfit stale data or slow training.

### `target_update_interval` and `tau`
These control how the target network is updated. A slower target update usually improves stability. SB3's default hard update behavior is often a good starting point.

## Practical Tuning Advice

For CartPole, a reasonable starting point is the default configuration in this project. If learning is too slow, try slightly higher `learning_rate` or more total timesteps. If learning is unstable, lower `learning_rate`, increase `buffer_size`, or keep exploration higher for longer.

If you moved to `Pendulum-v1`, the code structure would change significantly:
- The environment would become continuous-action.
- DQN would no longer be appropriate because it assumes a discrete action set.
- You would switch to DDPG, TD3, or SAC, where the policy outputs continuous actions directly.
- The training objective would involve actor and critic networks rather than a single Q-network over discrete actions.

## What SB3 Is Doing for You

Stable-Baselines3 supplies the Q-network, replay buffer, target network, epsilon-greedy exploration, optimization loop, and model serialization. That keeps this project focused on understanding the algorithm rather than rebuilding the entire reinforcement learning stack.
