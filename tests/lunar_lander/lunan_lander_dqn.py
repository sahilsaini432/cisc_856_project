import gymnasium as gym
from dqn_agent import DQNAgent
import tqdm as tqdm

env = gym.make("LunarLander-v3", render_mode="human")

# Setup agent
agent = DQNAgent(state_size=8, action_size=4)
batch_size = 64
episodes = 3000

for e in tqdm.tqdm(range(episodes)):
    observation, info = env.reset(seed=42)

    for time in range(2000):
        action = agent.act(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Update agent with experience
        agent.remember(observation, action, reward, next_observation, done)
        observation = next_observation

        if done and reward > -100:
            print(f"Episode: {e+1}/{episodes}, Score: {reward}, Epsilon: {agent.epsilon:.2f}")
            break

        agent.replay(batch_size)

    agent.decay_epsilon()
