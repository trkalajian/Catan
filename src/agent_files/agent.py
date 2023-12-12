from heuristics import build_settlement_heursitic

def q_learning(action_values, old_state, action, reward, new_state, alpha, gamma):
    new_action = greedy(action_values[new_state])
    action_values[old_state][action] += alpha * (
            reward + gamma * action_values[new_state][new_action] - action_values[old_state][action]
    )
    return None

class EpsilonGreedyDQNPolicy:
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

