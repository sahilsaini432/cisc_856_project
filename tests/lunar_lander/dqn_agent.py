import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Initialize neural network with 2 hidden layers
        Args:
            state_size: Dimension of input state
            action_size: Number of possible actions
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        """
        Forward pass through network
        Args:
            x: Input state tensor
        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        Initialize DQN agent with replay memory and hyperparameters
        Args:
            state_size: Dimension of state space
            action_size: Number of actions
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        # control how much future reward we care about (0 = only immediate, 1 = all future)
        self.gamma = 0.99
        # start with high exploration rate and decay over time
        # epsilon: Probability of choosing random action vs. model prediction
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        # increase the decay rate to make exploration last longer
        self.epsilon_decay = 0.997
        # Controls how much the model adjusts weights during training
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size)
        # For updating neural network weights during training.
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    # ////TODO: Update atc() to use something other than random action selection, like Boltzmann exploration or UCB, to encourage more exploration of less visited actions.//////
    def act(self, state):
        """
        Select action using epsilon-greedy policy
        Args:
            state: Current environment state
        Returns:
            Selected action index
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Train network on random batch from memory using Q-learning
        Args:
            batch_size: Number of experiences to sample
        """
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:

            # If the episode is done, the target is just the reward. Otherwise, we add the discounted max future reward.
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                # pass next state through model to get predicted Q-values for all actions
                next_q_values = self.model(next_state_tensor)
                # get the maximum predicted Q-value for the next state and use it to set the target reward
                max_q_value = torch.max(next_q_values).item()
                # update target to be the reward plus discounted max future reward
                target = reward + self.gamma * max_q_value

            # predict the current Q-values for the state
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_tensor)
            target_f[0][action] = target

            # Train Network
            # Clears previous gradients from last training step, PyTorch accumulates gradients by default, so must reset
            self.optimizer.zero_grad()

            # Mean Squared Error between predicted Q-values and target Q-values
            # Current prediction => self.model(state_tensor)
            # Desired output => target_f (which has the target Q-value for the action we took)
            # loss =? how wrong is the prediction
            loss = nn.MSELoss()(self.model(state_tensor), target_f)

            # back propagate the loss through the network to compute gradients
            loss.backward()

            # Update the network weights using calculated gradients
            self.optimizer.step()

    def decay_epsilon(self):
        # Only decay epsilon if the agent is performing reasonably well
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
