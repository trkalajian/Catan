import torch
import torch.nn as nn
import torch.optim as optim
from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import random

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the DQN network
input_size = 19 * (len(Resource) + 2) + 54 * 2  # Adjust based on your input features
output_size = 54  # Number of corner spots
dqn = DQNNetwork(input_size, output_size)


class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        if len(self.memory) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Initialize experience replay buffer
experience_replay = ExperienceReplay(capacity=500)

def mask_actions(valid_actions, action_values):
    masked_values = action_values.clone()
    masked_values[~valid_actions] = float('-inf')
    return masked_values

