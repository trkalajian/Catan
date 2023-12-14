# from heuristics import build_settlement, place_settlement
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def q_learning(action_values, old_state, action, reward, new_state, alpha, gamma):
    new_action = greedy(action_values[new_state])
    action_values[old_state][action] += alpha * (
            reward + gamma * action_values[new_state][new_action] - action_values[old_state][action]
    )
    return None

class EpsilonGreedyPolicy:
    def __init__(self, q_network, initial_epsilon, min_epsilon, random_state=None):
        self.q_network = q_network
        self.epsilon = initial_epsilon
        self.decay_rate = (initial_epsilon - min_epsilon) / 3000
        self.min_epsilon = min_epsilon
        self.random_state = random_state or np.random.RandomState(seed=0)

    def explore(self):
        return self.random_state.binomial(n=1, p=self.epsilon) == 1

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

    def __call__(self, state):
        state_tensor = torch.from_numpy(np.array(state, dtype=np.float32))

        if self.explore():
            print("I explored!")
            return POSSIBLE_ACTIONS[self.random_state.choice(range(self.q_network.fc2.out_features))]
        else:
            q_values = self.q_network(state_tensor)
            return POSSIBLE_ACTIONS[int(torch.argmax(q_values))]


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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

