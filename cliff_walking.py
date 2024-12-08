import gymnasium as gym

from rl.td import sarsa

env = gym.make("CliffWalking-v0")
num_states = env.unwrapped.observation_space.n
num_actions = env.unwrapped.action_space.n
discount_factor = 0.99
policy = sarsa(env, num_states, num_actions, discount_factor)

env.unwrapped.render_mode = "human"
observation, info = env.reset()
done = False
while not done:
    action = policy[observation]
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()
