import gymnasium as gym

from rl.mc import monte_carlo_off_policy


def build_state_indices(observation_space):
    state_indices = {}
    i = 0
    for player_sum in range(observation_space[0].n):
        for dealer_value in range(observation_space[1].n):
            for useable_ace in range(observation_space[2].n):
                state_indices[(player_sum, dealer_value, useable_ace)] = i
                i += 1
    return state_indices


def print_step(action, observation, reward):
    s = ""

    s += "State: ("
    s += f"Player Sum: {observation[0]}, "
    s += f"Dealer Value: {observation[1]}, "
    s += f"Usable Ace: {observation[2]}"
    s += "); "

    s += "Action: "
    if action == 0:
        s += "Stick"
    else:
        s += "Hit"
    s += "; "

    s += f"Reward: {reward}"

    print(s)


env = gym.make("Blackjack-v1", sab=True)

state_indices = build_state_indices(env.unwrapped.observation_space)
num_states = len(state_indices)
num_actions = env.unwrapped.action_space.n
discount_factor = 1.0
env_train = gym.wrappers.TransformObservation(
    env, lambda s: state_indices[s], gym.spaces.Discrete(num_states)
)
policy = monte_carlo_off_policy(env_train, num_states, num_actions, discount_factor)

observation, info = env.reset()
done = False
while not done:
    action = policy[state_indices[observation]]
    next_observation, reward, terminated, truncated, info = env.step(action)
    print_step(action, observation, reward)
    observation = next_observation
    done = terminated or truncated
env.close()
