import torch
import torch.nn as nn
import torch.optim as optim
from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import random
import numpy as np
import torch.nn.functional as F

POSSIBLE_ACTIONS = np.arange(start=0, stop=7)

def q_learning(action_values, old_state, action, reward, new_state, alpha, gamma):
    new_action = greedy(action_values[new_state])
    action_values[old_state][action] += alpha * (
            reward + gamma * action_values[new_state][new_action] - action_values[old_state][action]
    )
    return None

def greedy(state_action_values):
    max_actions = [action for action in state_action_values if
                   state_action_values[action] == max(state_action_values.values())]
    return random.choice(max_actions)

class EpsilonGreedyPolicy:
    # Define action constants
    BUILD_SETTLEMENT = 0
    BUILD_CITY = 1
    BUILD_ROAD = 2
    BUILD_CARD = 3
    MAKE_TRADE = 4
    PLAY_KNIGHT = 5
    PASS_TURN = 6

    def __init__(self, q_network, initial_epsilon, min_epsilon, decay_rate, random_state=None):
        self.q_network = q_network
        self.epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

    def explore(self):
        return self.random_state.binomial(n=1, p=self.epsilon) == 1

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def __call__(self, state, game, player):
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))

        # Determine valid actions
        validActions = []
        if player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) and game.board.get_valid_settlement_coords(player):
            validActions.append(self.BUILD_SETTLEMENT)
        if player.has_resources(BuildingType.CITY.get_required_resources()) and game.board.get_valid_city_coords(player):
            validActions.append(self.BUILD_CITY)
        if player.has_resources(BuildingType.ROAD.get_required_resources()) and game.board.get_valid_road_coords(player) and player.num_roads > 0:
            validActions.append(self.BUILD_ROAD)
        if player.has_resources(DevelopmentCard.get_required_resources()) and game.development_card_deck:
            validActions.append(self.BUILD_CARD)
        if player.get_possible_trades():
            validActions.append(self.MAKE_TRADE)
        if DevelopmentCard.KNIGHT in [card for card, amount in player.development_cards.items() if amount > 0]:
            validActions.append(self.PLAY_KNIGHT)
        validActions.append(self.PASS_TURN)

        # Choose action
        if self.explore():
            print("I explored!")
            pass_penalty = False
            action = random.choice(validActions)
            if action == 6 and len(validActions) > 1:
                pass_penalty = True
            return action, pass_penalty
        else:
            q_values = self.q_network(state_tensor)
            # Initialize masked_q_values with '-inf'
            masked_q_values = torch.full_like(q_values, float('-inf'))

            # Update masked_q_values only for valid actions
            for action in validActions:
                masked_q_values[action] = q_values[action]

            # Select the action with the highest q-value among valid actions
            action = POSSIBLE_ACTIONS[int(torch.argmax(masked_q_values))]
            pass_penalty = False
            if action == 6 and len(validActions)>1:
                pass_penalty = True
            return action, pass_penalty


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
# experience_replay = ExperienceReplay(capacity=500)

def mask_actions(valid_actions, action_values):
    masked_values = action_values.clone()
    masked_values[~valid_actions] = float('-inf')
    return masked_values

