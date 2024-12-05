import gymnasium as gym
import numpy as np


def monte_carlo_exploring_starts(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    max_episodes: int = 1_000,
):
    pi = np.random.randint(0, num_actions - 1, size=num_states, dtype=np.int_)
    Q = np.zeros((num_states, num_actions))
    N = np.zeros((num_states, num_actions), dtype=np.int_)

    for _ in range(max_episodes):
        episode = []
        s, _ = env.reset()
        done = False
        while not done:
            a = pi[s]
            s_next, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_next
            done = terminated or truncated
        G = 0.0
        for s, a, r in reversed(episode):
            G = discount_factor * G + r
            if N[s, a] != 0:
                continue
            N[s, a] += 1
            Q[s, a] += (G - Q[s, a]) / N[s, a]
            pi[s] = np.argmax(Q[s, :])

    return pi
