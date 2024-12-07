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


def monte_carlo_epsilon_soft(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    epsilon: float = 0.01,
    max_episodes: int = 1_000,
):
    # TODO: Handle cases where the number of actions depends on the state

    pi = np.ones((num_states, num_actions)) / num_actions
    Q = np.zeros((num_states, num_actions))
    N = np.zeros((num_states, num_actions), dtype=np.int_)

    for _ in range(max_episodes):
        episode = []
        s, _ = env.reset()
        done = False
        while not done:
            a = np.random.choice(range(num_actions), p=pi[s, :])
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
            a_greedy = np.argmax(Q[s, :])
            for a in range(num_actions):
                if a == a_greedy:
                    pi[s, a] = 1.0 - epsilon + epsilon / num_actions
                else:
                    pi[s, a] = epsilon / num_actions

    return np.argmax(pi, axis=1)


def monte_carlo_off_policy(
    env: gym.Env,
    num_states: int,
    num_actions: int,
    discount_factor: float,
    max_episodes: int = 1_000,
):
    Q = np.zeros((num_states, num_actions))
    C = np.zeros((num_states, num_actions))
    pi = np.argmax(Q, axis=1)

    for _ in range(max_episodes):
        b = np.ones((num_states, num_actions)) / num_actions
        episode = []
        s, _ = env.reset()
        done = False
        while not done:
            a = np.argmax(b[s, :])
            s_next, r, terminated, truncated, _ = env.step(a)
            episode.append((s, a, r))
            s = s_next
            done = terminated or truncated
        G = 0.0
        W = 1.0
        for s, a, r in reversed(episode):
            G = discount_factor * G + r
            C[s, a] += W
            Q[s, a] += W / C[s, a] * (G - Q[s, a])
            pi[s] = np.argmax(Q[s, :])
            if a != pi[s]:
                break
            W *= 1 / b

    return pi
