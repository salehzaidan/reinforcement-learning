import gymnasium as gym
import numpy as np

from rl.dp import value_iteration


def visualize_policy(policy: np.ndarray, num_states: int):
    size = int(np.sqrt(num_states))
    arrows = ["←", "↓", "→", "↑"]
    for row in policy.reshape(size, size):
        print(" ".join(arrows[col] for col in row))


env = gym.make("FrozenLake-v1", render_mode="human")

num_states = env.unwrapped.observation_space.n
num_actions = env.unwrapped.action_space.n
transition_prob = np.zeros((num_states, num_actions, num_states))
reward_fn = np.zeros((num_states, num_actions, num_states))
for s in range(num_states):
    for a in range(num_actions):
        for p, s_next, r, _ in env.unwrapped.P[s][a]:
            transition_prob[s, a, s_next] += p
            reward_fn[s, a, s_next] += r
discount_factor = 0.99
policy = value_iteration(
    num_states, num_actions, transition_prob, reward_fn, discount_factor
)
visualize_policy(policy, num_states)

observation, info = env.reset()
done = False
while not done:
    action = policy[observation]
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
