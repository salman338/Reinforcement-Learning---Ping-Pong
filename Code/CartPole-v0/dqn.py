import math
import random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        
        self.eps = self.eps_start
        self.steps = 0

    def forward(self, x):
        """
        Runs the forward pass of the NN depending on architecture.
        """
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

    def act(self, observation, exploit=False):
        """
        Selects an action with an epsilon-greedy exploration strategy.
        """

        # Implement action selection using the Deep Q-network. This function
        # takes an observation tensor and should return a tensor of actions.

        self.steps = self.steps + 1
        self.eps = self.eps_end + (self.eps_start - self.eps_end) *  math.exp(-1. * self.steps / self.anneal_length)

        if self.eps < self.eps_end:
            self.eps = self.eps_end
        
        ran_num = random.random()

        # Implement epsilon-greedy exploration.
        if ran_num < self.eps:
            actions = [random.randint(0,self.n_actions-1)]
            actions =  torch.tensor(actions)
            
            return actions
        else:
            self.eval()
            
            with torch.no_grad():
                q_values = self.forward(observation)
            
            self.train()
            actions = torch.tensor([torch.argmax(q_values)])

            # raise NotImplmentedError
            return actions

def optimize(dqn, target_dqn, memory, optimizer):
    """
    This function samples a batch from the replay buffer and optimizes the Q-network.
    """
    
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return 0

    # Sample a batch from the replay memory and concatenate 
    observations, actions, next_observations, rewards, dones = memory.sample(dqn.batch_size)

    observations = torch.stack(list(observations), dim=0)
    actions = torch.stack(list(actions), dim=0)
    rewards = torch.stack(list(rewards), dim=0)
    next_observations =  torch.stack(list(next_observations), dim=0)
    dones =  torch.stack(list(dones), dim=0)

    # Compute the current estimates of the Q-values for each state-action
    # pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    # corresponding to the chosen actions.
    q_values = dqn(observations)
    params = list(dqn.parameters())
    q_values =  torch.gather(q_values, 1, (actions.unsqueeze(-1)).long())

    # Compute the Q-value targets. Only do this for non-terminal transitions!
    next_q_value_targets = target_dqn(next_observations).detach().max(1)[0]
    q_value_targets = rewards + (target_dqn.gamma * next_q_value_targets * (1 - dones))

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
