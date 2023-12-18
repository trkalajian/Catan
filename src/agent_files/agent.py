# from heuristics import build_settlement, place_settlement
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from agent_files.DQN import DQNNetwork, ExperienceReplay, EpsilonGreedyPolicy
from features import Features  # Import the Features class
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# class to initialize the heuristic agent
class HeuristicAgent:
    def __init__(self, policy, place_settlement_func, place_road_func, place_robber_func, choose_best_trade, place_city_func):
        self.policy = policy
        self.place_settlement_func = place_settlement_func
        self.place_road = place_road_func
        self.place_robber = place_robber_func
        self.choose_trade_func = choose_best_trade
        self.place_city_func = place_city_func

    def choose_action(self, game, current_player_num, is_start):
        # Implement your heuristic policy logic here
        action = self.policy(game, current_player_num, is_start)
        return action

    def place_settlement(self, game, current_player_num, is_start):
        # Call the heuristic function to choose the settlement location
        settlement_coords = self.place_settlement_func(game, current_player_num, is_start)
        return settlement_coords  # Return the chosen settlement coordinates

    def place_road(self, game, current_player_num):
        road_coords = self.place_road(game, current_player_num)
        return road_coords

    def place_robber(self, game, current_player_num):
        robber_coords = self.place_robber(game, current_player_num)
        return robber_coords

    def choose_trade_func(self, game, current_player_num, possible_trades):
        trade = self.choose_trade_func(game, current_player_num, possible_trades)
        return trade

    def place_city_func(self, game, valid_city_coords):
        city_coords = self.place_city_func(game, valid_city_coords)
        return city_coords


class DQNAgent(HeuristicAgent):
    # Initialize the DQN Agent
    def __init__(self, input_size, output_size, place_settlement_func, place_road_func, place_robber_func,
                 choose_best_trade, place_city_func, memory_size=500, batch_size=32, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        super().__init__(self.dqn_policy, place_settlement_func, place_road_func, place_robber_func, choose_best_trade,
                         place_city_func)

        # Initialize the main model and the target model for DQN
        self.model = DQNNetwork(input_size, output_size)
        self.target_model = DQNNetwork(input_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize experience replay for storing transitions
        self.experience_replay = ExperienceReplay(memory_size)

        # Define the epsilon-greedy policy for action selection
        self.policy = EpsilonGreedyPolicy(self.model, epsilon, epsilon_min, epsilon_decay)

        # Initialize training parameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    # Define the policy to use for the DQN agent
    def dqn_policy(self, game, current_player_num):
        state = self.get_state(game, current_player_num)
        action_number, pass_penalty = self.policy(state, game, game.players[current_player_num])

        # Map the action number to the corresponding game action
        action_mapping = {
            0: [1, 1],
            1: [1, 2],
            2: [1, 3],
            3: [1, 4],
            4: [2, None],
            5: [3, None],
            6: [4, None]
        }

        return action_mapping.get(action_number, [4, None]), pass_penalty  # Default action if action_number is not in the mapping

    # Get the current state representation for the DQN input
    def get_state(self, game, current_player_num):
        features = Features(game, game.players[current_player_num])
        state = features.flattenFeature(game, game.players[current_player_num])
        return np.array(state, dtype=np.float32)

    # Training step for the DQN agent
    def train(self):
        # Check if the memory is sufficient for training
        if len(self.experience_replay.memory) < self.batch_size:
            return
        transitions = self.experience_replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert each state, action, and reward in the batch to a Tensor
        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32) for s in batch.state])
        action_batch = torch.tensor([a for a in batch.action], dtype=torch.long)
        action_batch = action_batch.unsqueeze(1)
        reward_batch = torch.tensor([r for r in batch.reward], dtype=torch.float32)

        # Create a mask of non-final states
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.stack(
            [torch.tensor(s, dtype=torch.float32) for s in batch.next_state if s is not None])

        # Compute current Q values
        state_action_values = self.model(state_batch).gather(1, action_batch)

        # Compute next Q values from the target network for non-final states
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute the loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for the epsilon-greedy policy
        self.policy.decay_epsilon()

    # Update the target network with weights from the main model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


