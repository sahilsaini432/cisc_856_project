import gymnasium as gym
from stable_baselines3 import A2C
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set up the environment
env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8), is_slippery=True, render_mode="human")

# Initialize A2C model
model = A2C("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Test the trained model
obs, info = env.reset()

for step in tqdm(range(1000)):
    action, _states = model.predict(obs, deterministic=True)
    print(f"\n--- Step {step + 1} ---")
    print(f"Current observation: {obs}")
    print(f"Action taken: {action}")
    print(f"State info: {_states}")

    obs, reward, terminated, truncated, info = env.step(int(action))
    print(f"Next observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    if terminated or truncated:
        print(f"Episode ended at step {step + 1}")
        obs, info = env.reset()
        print(f"Environment reset. New observation: {obs}")
