import gymnasium as gym
import numpy as np

__all__ = ["sarsa", "q_learning", "expected_sarsa", "double_q_learning", "n_step_sarsa"]


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


def n_step_sarsa(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    epsilon: float = 0.01,
    alpha: float = 0.1,
    n: int = 2,
    max_episodes: int = 1_000,
):
    pi = np.ones((num_states, num_actions)) / num_actions
    Q = np.zeros((num_states, num_actions))

    for _ in range(max_episodes):
        S = []
        A = []
        R = []
        s, _ = env.reset()
        S.append(s)
        a = np.random.choice(range(num_actions), p=pi[s, :])
        A.append(a)
        T = np.inf
        t = 0
        while True:
            if t < T:
                a = A[-1]
                s_next, r, terminated, truncated, _ = env.step(a)
                S.append(s_next)
                R.append(r)
                if terminated or truncated:
                    T = t + 1
                else:
                    a_next = np.random.choice(range(num_actions), p=pi[s_next, :])
                    A.append(a_next)
                s = s_next
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + n, T)):
                    G += discount_factor ** (i - tau - 1) * R[i]
                if tau + n < T:
                    s_n = S[tau + n]
                    a_n = A[tau + n]
                    G += discount_factor**n * Q[s_n, a_n]
                s = S[tau]
                a = A[tau]
                Q[s, a] += alpha * (G - Q[s, a])

                a_greedy = np.argmax(Q[s, :])
                for a in range(num_actions):
                    if a == a_greedy:
                        pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                    else:
                        pi[s, a] = epsilon / num_actions

            if tau == T - 1:
                break
            t += 1

    return np.argmax(pi, axis=1)
