import numpy as np


def policy_iteration(
    num_states: int,
    num_actions: int,
    transition_prob: np.ndarray,
    reward_fn: np.ndarray,
    discount_factor: float,
    theta: float = 1e-6,
):
    V = np.zeros(num_states)
    pi = np.random.randint(0, num_actions - 1, size=num_states, dtype=np.int_)

    while True:
        while True:
            delta = 0.0
            for s in range(num_states):
                v_old = V[s]
                v = 0.0
                for s_next in range(num_states):
                    p = transition_prob[s, pi[s], s_next]
                    r = reward_fn[s, pi[s], s_next]
                    v += p * (r + discount_factor * V[s_next])
                V[s] = v
                delta = max(delta, abs(v_old - v))
            if delta < theta:
                break

        stable = True
        for s in range(num_states):
            a_old = pi[s]
            Q = np.zeros(num_actions)
            for a in range(num_actions):
                Q[a] = 0.0
                for s_next in range(num_states):
                    p = transition_prob[s, a, s_next]
                    r = reward_fn[s, a, s_next]
                    Q[a] += p * (r + discount_factor * V[s_next])
            pi[s] = np.argmax(Q)
            if a_old != pi[s]:
                stable = False
                break
        if stable:
            return pi