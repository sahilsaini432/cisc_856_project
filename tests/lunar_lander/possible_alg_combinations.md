- DQN agent using epsilon-greedy exploration and experience replay
## Modifications for DQN Agent
1. Double DQN (DDQN)
Add target network (separate from main network)
Reduces overestimation of Q-values

2. Dueling DQN
Split network into value and advantage streams
Better learns state values

3. Prioritized Experience Replay
Sample important experiences more frequently
Weight by TD error

4. Noisy Networks
Replace epsilon-greedy with learned exploration noise
Add noise layers to network

5. Rainbow DQN
Combines: DDQN + Dueling + Prioritized Replay + Noisy Nets + Multi-step learning + Distributional RL

6. Hyperparameter tuning:
Network size (layers, neurons)
Batch size
Memory size
Learning rate
Gamma/epsilon values

# Other Algorithms that might be good for Lunar Lander Env
Policy-Based:
1. PPO (Proximal Policy Optimization)

Most popular for continuous/discrete actions
Stable, easy to tune
2. A2C/A3C (Advantage Actor-Critic)

Actor-critic architecture
Good baseline
3. TRPO (Trust Region Policy Optimization)

Similar to PPO, more complex
Value-Based:

4. SARSA

On-policy alternative to Q-learning
More conservative
5. Monte Carlo methods

Simple, episodic learning
Hybrid:

6. SAC (Soft Actor-Critic)

State-of-the-art for continuous control
Maximum entropy framework
7. TD3 (Twin Delayed DDPG)

Improved DDPG with twin critics
Model-Based:

8. World Models

Learn environment model, plan in latent space
9. MuZero

Combines planning + learning without model