# import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from collections import deque, namedtuple


device = "cpu"
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self, action_space, observation_space):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.action_space = action_space
        self.observation_space = observation_space

        state_size = 8
        action_size = 4
        seed = 42
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        # self.t_step = 0
    
    # def step(self, state, action, reward, next_state, done):
    #     # Save experience in replay memory
    #     self.memory.add(state, action, reward, next_state, done)
        
    #     # Learn every UPDATE_EVERY time steps.
    #     self.t_step = (self.t_step + 1) % UPDATE_EVERY
    #     if self.t_step == 0:
    #         # If enough samples are available in memory, get random subset and learn
    #         if len(self.memory) > BATCH_SIZE:
    #             experiences = self.memory.sample()
    #             self.learn(experiences, GAMMA)

    def act(self, observation, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        self.state = observation
        state = observation
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            self.action =  np.argmax(action_values.cpu().data.numpy())
        else:
            self.action =  random.choice(np.arange(self.action_size))
        
        return self.action

    def learn(self, observation, reward, terminated, truncated):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        # Obtain random minibatch of tuples from D
        state, action, next_state, done = self.state, self.action, observation, (terminated or truncated)

        ## Compute and minimize the loss
        ### Extract next maximum estimated value from target network
        q_targets_next = self.qnetwork_target(next_state).max()
        # .detach().max(1)[0].unsqueeze(1)
        ### Calculate target value from bellman equation
        q_targets = reward + GAMMA * q_targets_next * (1 - done)
        ### Calculate expected value from local network
        # print('target_next:')
        # print(q_targets_next)
        # print('target:')
        # print(q_targets)
        # print('expected:')
        # print(self.qnetwork_local(state))
        # print('actions:')
        # print(action)

        q_expected = self.qnetwork_local(state)[action]
        
        ### Loss calculation (we used Mean squared error)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# Define the neural network
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(torch.Tensor(state))
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)