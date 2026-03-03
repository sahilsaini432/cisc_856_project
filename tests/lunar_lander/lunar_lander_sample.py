import gymnasium as gym

# 1. Setup the environment
# render_mode="human" allows you to see the game window
env = gym.make("LunarLander-v3", render_mode="human")

# 2. Reset to start (returns initial observation and info)
observation, info = env.reset(seed=42)

for _ in range(1000):
    # 3. Choose an action (sampling randomly for now)
    action = env.action_space.sample()

    # 4. Apply action and get stats
    # observation: The state (e.g., position, velocity)
    # reward: The score for that step
    # terminated: Did the agent win or crash?
    # truncated: Did the time limit run out?
    observation, reward, terminated, truncated, info = env.step(action)

    # 5. Handle end of episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
