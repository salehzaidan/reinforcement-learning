import gymnasium as gym
import numpy as np


def sarsa(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    epsilon: float = 0.01,
    alpha: float = 0.1,
    max_episodes: int = 1_000,
):
    pi = np.ones((num_states, num_actions)) / num_actions
    Q = np.zeros((num_states, num_actions))

    for _ in range(max_episodes):
        s, _ = env.reset()
        a = np.random.choice(range(num_actions), p=pi[s, :])
        done = False
        while not done:
            s_next, r, terminated, truncated, _ = env.step(a)
            a_next = np.random.choice(range(num_actions), p=pi[s_next, :])
            Q[s, a] += alpha * (r + discount_factor * Q[s_next, a_next] - Q[s, a])
            a_greedy = np.argmax(Q[s, :])
            for a in range(num_actions):
                if a == a_greedy:
                    pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                else:
                    pi[s, a] = epsilon / num_actions
            s = s_next
            a = a_next
            done = terminated or truncated

    return np.argmax(pi, axis=1)
