import gymnasium as gym
import numpy as np

__all__ = ["sarsa", "q_learning", "expected_sarsa", "double_q_learning"]


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


def q_learning(
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
        done = False
        while not done:
            a = np.random.choice(range(num_actions), p=pi[s, :])
            s_next, r, terminated, truncated, _ = env.step(a)
            Q[s, a] += alpha * (r + discount_factor * np.max(Q[s_next, :]) - Q[s, a])
            a_greedy = np.argmax(Q[s, :])
            for a in range(num_actions):
                if a == a_greedy:
                    pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                else:
                    pi[s, a] = epsilon / num_actions
            s = s_next
            done = terminated or truncated

    return np.argmax(Q, axis=1)


def expected_sarsa(
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
        done = False
        while not done:
            a = np.random.choice(range(num_actions), p=pi[s, :])
            s_next, r, terminated, truncated, _ = env.step(a)
            Q[s, a] += alpha * (
                r + discount_factor * np.sum(pi[s_next, :] * Q[s_next, :]) - Q[s, a]
            )
            a_greedy = np.argmax(Q[s, :])
            for a in range(num_actions):
                if a == a_greedy:
                    pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                else:
                    pi[s, a] = epsilon / num_actions
            s = s_next
            done = terminated or truncated

    return np.argmax(Q, axis=1)


def double_q_learning(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    epsilon: float = 0.01,
    alpha: float = 0.1,
    max_episodes: int = 1_000,
):
    pi = np.ones((num_states, num_actions)) / num_actions
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))

    for _ in range(max_episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = np.random.choice(range(num_actions), p=pi[s, :])
            s_next, r, terminated, truncated, _ = env.step(a)
            if np.random.random() < 0.5:
                Q1[s, a] += alpha * (
                    r
                    + discount_factor * Q2[s_next, np.argmax(Q1[s_next, :])]
                    - Q1[s, a]
                )
            else:
                Q2[s, a] += alpha * (
                    r
                    + discount_factor * Q1[s_next, np.argmax(Q2[s_next, :])]
                    - Q2[s, a]
                )

            a_greedy = np.argmax(Q1[s, :] + Q2[s, :])
            for a in range(num_actions):
                if a == a_greedy:
                    pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                else:
                    pi[s, a] = epsilon / num_actions
            s = s_next
            done = terminated or truncated

    return np.argmax(pi, axis=1)
