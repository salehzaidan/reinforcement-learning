import argparse
import gymnasium as gym

import rl.td


def get_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo",
        choices=rl.td.__all__.extend(rl.td.__all__),
        default="sarsa",
    )
    return parser.parse_args()


def main():
    args = get_command_line_args()
    if args.algo == "sarsa":
        print("Using Sarsa")
        algo = rl.td.sarsa
    elif args.algo == "q_learning":
        print("Using Q-Learning")
        algo = rl.td.q_learning
    elif args.algo == "expected_sarsa":
        print("Using Expected Sarsa")
        algo = rl.td.expected_sarsa

    env = gym.make("CliffWalking-v0")
    num_states = env.unwrapped.observation_space.n
    num_actions = env.unwrapped.action_space.n
    discount_factor = 0.99
    policy = algo(env, num_states, num_actions, discount_factor)

    env.unwrapped.render_mode = "human"
    observation, info = env.reset()
    done = False
    while not done:
        action = policy[observation]
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    env.close()


if __name__ == "__main__":
    main()
