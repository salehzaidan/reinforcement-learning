import gymnasium as gym


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


env = gym.make("Blackjack-v1", render_mode="human")
observation, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print_step(action, observation, reward)
    done = terminated or truncated
env.close()
