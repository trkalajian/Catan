from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
from methods import choose_path, choose_intersection, choose_resource, move_robber, count_cards, resource_check, \
    choose_hex, get_coord_sort_by_xy
from colored import fore_rgb, attr, stylize
from agent_files.agent import HeuristicAgent, DQNAgent
from agent_files.heuristics import place_settlement, heuristic_policy, choose_road_placement, \
    place_robber, choose_best_trade, place_city
import actorcritic
import random
import pickle
import os
from training_visualization import TrainingVisualizer
from features import Features
import numpy as np
from actorcritic import ActorCritic
import matplotlib.pyplot as plt

# set number of players here
numAgents = 4
num_iterations = 10
num_episodes = 200
save_every = 100
all_rewards = np.zeros((num_iterations, num_episodes, numAgents))
all_loss = np.zeros((num_iterations, num_episodes))
final_vps = np.zeros((num_iterations, num_episodes, numAgents))
winsPerPlayer = np.zeros(numAgents)
visualizer = TrainingVisualizer()


# Function to create a new board and game
def create_new_game():
    board = BeginnerBoard()
    # BoardRenderer.player_color_map = {}
    # BoardRenderer._unused_player_colors = BoardRenderer.DEFAULT_PLAYER_COLORS
    game = Game(board, numAgents)
    renderer = BoardRenderer(game.board, {})
    return game, renderer


def translate_action_to_index(action_pair):
    action_mapping = {
        (1, 1): 0,
        (1, 2): 1,
        (1, 3): 2,
        (1, 4): 3,
        (2, None): 4,
        (3, None): 5,
        (4, None): 6
    }
    action_pair_tuple = tuple(action_pair)  # Convert the action pair list to a tuple
    return action_mapping.get(action_pair_tuple, 6)  # Default to 'PASS_TURN' if action_pair not in mapping


def save_progress(agents, all_rewards, final_vps, episode, iteration):
    """
    Saves the policies of non-Heuristic agents, all_rewards, and final_vps.
    """
    # Creating directories if they don't exist
    os.makedirs('results/theta', exist_ok=True)
    os.makedirs('results/dqn', exist_ok=True)
    os.makedirs('results/rewards', exist_ok=True)
    os.makedirs('results/final_vps', exist_ok=True)

    # Iterate through agents and save their policies if they are not exactly HeuristicAgents
    for i, agent in enumerate(agents):
        if type(agent) is not HeuristicAgent:
            policy_type = 'theta' if hasattr(agent, 'theta') else 'dqn'
            policy_filename = f'results/{policy_type}/{policy_type}_policy_{episode + 1}_{iteration + 1}.pkl'
            with open(policy_filename, 'wb') as f:
                policy_data = agent.theta if hasattr(agent, 'theta') else agent.model.state_dict()
                pickle.dump(policy_data, f)
            print(f"Saved policy for Player {i} ({policy_type}) at episode {episode + 1}, iteration {iteration + 1}")

    np.savetxt(f'results/rewards/{iteration}iteration_rewards_{episode + 1}.csv', all_rewards[iteration], delimiter=',')
    np.savetxt(f'results/final_vps/{iteration}iteration_vps_{episode + 1}.csv', final_vps[iteration], delimiter=',')
    print(f"Rewards and victory points saved at episode {episode + 1}, iteration {iteration + 1}")


# function to calculate the reward from the most recent distribution of resources
def resource_reward(game, previous_total_resources):
    card_diff = np.zeros(numAgents)
    for i in range(numAgents):
        cur_player = game.players[i]
        card_diff[i] = sum(cur_player.resources.values()) - previous_total_resources[i]
    return card_diff * 0.1


# function calculates the final victory point rewards at the end of a game (episode)
def final_vp_reward(game):
    max_vps = max(game.get_victory_points(p) for p in game.players)
    vp_rewards = np.array([game.get_victory_points(p) for p in game.players])
    winner_indices = [i for i, p in enumerate(game.players) if game.get_victory_points(p) == max_vps]
    for index in winner_indices:
        vp_rewards[index] += 100
    return vp_rewards


def process_dqn_step(agent, game, current_player_num, turn_reward, current_state, done=False):
    """
    Process a single step for the DQN agent.
    """
    current_player = game.players[current_player_num]

    if isinstance(agent, DQNAgent):
        next_feats = Features(game, current_player)
        next_state = next_feats.flattenFeature(game, current_player)
        agent.experience_replay.push(current_state, num_choice, next_state, turn_reward, done=done)

        return next_state


#  method to build an agent that runs solely using heuristics
def heuristic_agent_maker():
    heur_policy = heuristic_policy
    return HeuristicAgent(
        policy=heur_policy,
        place_settlement_func=place_settlement,
        place_road_func=choose_road_placement,
        place_robber_func=place_robber,
        choose_best_trade=choose_best_trade,
        place_city_func=place_city
    )


# method to build an agent that runs DQN. Contains all hyperparameters and heuristics
def DQN_agent_maker():
    return DQNAgent(input_size=98,  # number of features in features.py after flattening
                    output_size=7,  # number of available actions
                    place_settlement_func=place_settlement,
                    place_road_func=choose_road_placement,
                    place_robber_func=place_robber,
                    choose_best_trade=choose_best_trade,
                    place_city_func=place_city,
                    memory_size=500,
                    batch_size=32,
                    gamma=0.99,
                    epsilon=1.0,
                    epsilon_min=0.01,
                    epsilon_decay=0.99,
                    learning_rate=0.001
                    )


# method to implement actor critic
def actor_critic_maker():
    return actorcritic.ActorCritic(place_settlement_func=place_settlement,
                                   place_road_func=choose_road_placement,
                                   place_robber_func=place_robber,
                                   choose_best_trade=choose_best_trade,
                                   place_city_func=place_city
                                   )


# # Main game loop
current_player_num = 0
is_start = False
num_games = 0
num_turns = 0
for iteration in range(num_iterations):
    loss = []
    agents = []
    for j in range(numAgents):
        # change some of these to actor-critics when appropriate
        if j >= 1:
            agents.append(heuristic_agent_maker())
        else:
            # creates an actor critic
            # agents.append(actor_critic_maker())
            agents.append(DQN_agent_maker())

    # Main game loop
    for ep in range(num_episodes):
        print("Game Number: " + str(num_games + 1))
        total_reward = np.zeros(numAgents)  # Initialize total reward for this episode
        # Create a new board and game for each episode
        game, renderer = create_new_game()
        choices = []
        # setting the order
        player_order = list(range(len(game.players)))
        random.shuffle(player_order)

        for i in player_order + list(reversed(player_order)):
            is_start = True
            current_player = game.players[i]
            if type(agents[i]) is actorcritic.ActorCritic:
                agents[i].initializeEpisode(game, current_player)
            # coords = choose_intersection(game.board.get_valid_settlement_coords(current_player, ensure_connected=False),
            #                              "Where do you want to build your settlement? ")
            agent_coords = agents[i].place_settlement(game, i, is_start)
            game.build_settlement(player=current_player, coords=agent_coords, cost_resources=False,
                                  ensure_connected=False)
            current_player.add_resources(game.board.get_hex_resources_for_intersection(agent_coords))
            # road_options = game.board.get_valid_road_coords(current_player, connected_intersection=agent_coords)
            road_coords = agents[i].place_road(game, i, is_start)
            game.build_road(player=current_player, path_coords=road_coords, cost_resources=False)

        print(game.board)
        player_colors = []
        current_player_num = player_order[0]
        dice = 0
        while True:
            if type(agents[current_player_num]) is DQNAgent:
                num_turns += 1
            current_player = game.players[current_player_num]
            current_feats = Features(game, current_player)
            current_state = current_feats.flattenFeature(game, game.players[
                current_player_num])  # get the current state before action
            turn_reward = 0
            # Roll the dice
            dice = random.randint(1, 6) + random.randint(1, 6)
            is_start = False
            if dice == 7:
                card_totals = count_cards(game)
                sev_reward = resource_check(card_totals,
                                            game)  # discards people over 7 and returns neg reward porportional to discard
                turn_reward += sev_reward[0]
                for i in range(numAgents):
                    total_reward[i] += sev_reward[i]
                hex_coords, player_stolen = agents[current_player_num].place_robber(game, current_player_num)
                move_robber(current_player, game, hex_coords, player_stolen)
                pass
            else:  # give out resources and rewards
                previous_total_resources = np.zeros(numAgents)
                for i in range(numAgents):
                    cur_player = game.players[i]
                    previous_total_resources[i] = sum(
                        cur_player.resources.values())  # Store the player's previous resources

                game.add_yield_for_roll(dice)  # add the yield for the dice roll

                resource_rewards = resource_reward(game, previous_total_resources)  # calculate each players reward
                turn_reward += resource_rewards[0]
                for i in range(numAgents):
                    total_reward[i] += resource_rewards[i]
            choice = [0, 0]
            while choice[0] != 4:
                while True:
                    try:
                        if type(agents[current_player_num]) is DQNAgent:
                            choice, pass_penalty = agents[current_player_num].dqn_policy(game, current_player_num)
                            num_choice = translate_action_to_index(choice)
                            if pass_penalty:
                                turn_reward += -10  # penalizes the player if they passed with available actions
                            choices.append(choice)
                        else:
                            choice = agents[current_player_num].policy(game, current_player_num)
                        print(choice)
                        if 1 <= choice[0] <= 4:
                            break  # Valid choice, exit the loop
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                if choice[0] == 1:
                    building_choice = choice[1]
                    if not 1 <= building_choice <= 5:
                        continue
                    if building_choice == 1:
                        if not current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
                            continue
                        valid_coords = game.board.get_valid_settlement_coords(current_player)
                        if not valid_coords:
                            continue
                        #print("Player %d is building a settlement" % (current_player_num + 1))
                        coords = agents[current_player_num].place_settlement(game, current_player_num, is_start)
                        game.build_settlement(current_player, coords)
                        if type(agents[current_player_num]) is DQNAgent:
                            turn_reward += 5  # turn reward for the VP earned from settlement
                            current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                             turn_reward, current_state, done=True)
                    elif building_choice == 2:
                        if not current_player.has_resources(BuildingType.CITY.get_required_resources()):
                            continue
                        valid_coords = game.board.get_valid_city_coords(current_player)
                        # coords = choose_intersection(valid_coords, "Where do you want to build a city?  ", game, renderer)
                        coords = agents[current_player_num].place_city_func(game, valid_coords)
                        #print("Player %d is building a city" % (current_player_num + 1))
                        game.upgrade_settlement_to_city(current_player, coords)
                        if type(agents[current_player_num]) is DQNAgent:
                            turn_reward += 5  # turn reward for the VP earned from the city
                            current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                             turn_reward, current_state, done=True)
                    elif building_choice == 3:
                        if not current_player.has_resources(BuildingType.ROAD.get_required_resources()):
                            continue
                        valid_coords = game.board.get_valid_road_coords(current_player)
                        if not valid_coords:
                            continue
                        path_coords = agents[current_player_num].place_road(game, current_player_num, is_start)
                        #print("Player %d is building a road" % (current_player_num + 1))
                        game.build_road(current_player, path_coords)
                        if type(agents[current_player_num]) is DQNAgent:
                            turn_reward += 2  # turn reward for the VP earned from road build
                            current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                                 turn_reward, current_state, done=True)
                    elif building_choice == 4:
                        if not current_player.has_resources(DevelopmentCard.get_required_resources()):
                            continue
                        #print("Player %d is buying a card" % (current_player_num + 1))
                        dev_card = game.build_development_card(current_player)
                        if type(agents[current_player_num]) is DQNAgent:
                            current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                             turn_reward, current_state, done=True)
                    elif building_choice == 5:
                        break

                elif choice[0] == 2:
                    possible_trades = list(current_player.get_possible_trades())
                    # trade_choice = int(input('->  '))
                    trade = agents[current_player_num].choose_trade_func(game, current_player_num, possible_trades)
                    #print("Player %d is trading" % (current_player_num + 1))
                    if trade == None:
                        print('woa')
                    current_player.add_resources(trade)
                    if type(agents[current_player_num]) is DQNAgent:
                        current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                         turn_reward, current_state, done=True)
                elif choice[0] == 3:
                    dev_cards = [card for card, amount in current_player.development_cards.items() if
                                 amount > 0 and card is not DevelopmentCard.VICTORY_POINT]
                    card_to_play = DevelopmentCard.KNIGHT
                    #print("Player %d is playing a knight" % (current_player_num + 1))
                    game.play_development_card(current_player, card_to_play)
                    if card_to_play is DevelopmentCard.KNIGHT:
                        hex_coords, player_stolen = agents[current_player_num].place_robber(game, current_player_num)
                        move_robber(current_player, game, hex_coords, player_stolen)
                    if type(agents[current_player_num]) is DQNAgent:
                        current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                         turn_reward, current_state, done=False)
                    # elif card_to_play is DevelopmentCard.YEAR_OF_PLENTY:
                    #     for _ in range(2):
                    #         resource = choose_resource("What resource do you want to receive?")
                    #         current_player.add_resources({resource: 1})
                    # elif card_to_play is DevelopmentCard.ROAD_BUILDING:
                    #     for _ in range(2):
                    #         valid_path_coords = game.board.get_valid_road_coords(current_player)
                    #         path_coords = choose_path(valid_path_coords, "Choose where to build a road: ")
                    #         game.build_road(current_player, path_coords, cost_resources=False)
                    # elif card_to_play is DevelopmentCard.MONOPOLY:
                    #     resource = choose_resource("What resource do you want to take?")
                    #     for i in range(len(game.players)):
                    #         player = game.players[i]
                    #         if player is not current_player:
                    #             amount = player.resources[resource]
                    #             player.remove_resources({resource: amount})
                    #             current_player.add_resources({resource: amount})
            if game.get_victory_points(current_player) >= 10:
                winsPerPlayer[current_player_num] += 1
                # Calculate final VP rewards for all players
                vp_rewards = final_vp_reward(game)
                turn_reward += vp_rewards[0]  # final rewards for the round
                for i in range(numAgents):
                    total_reward[i] += vp_rewards[i]
                    player_colors.append(renderer._get_player_color(game.players[i]))

                print("Current Victory point standings:")
                for i, player in enumerate(game.players):
                    player_color = renderer.player_color_map[game.players[i]]
                    vp = game.get_victory_points(game.players[i])
                    print(stylize(f"Player {i + 1}: {vp} VP",
                                  fore_rgb(player_color[0], player_color[1], player_color[2])))
                print(stylize("Current longest road owner: %s" % (
                    "Player %d" % (game.players.index(
                        game.longest_road_owner) + 1) if game.longest_road_owner else "Nobody"), attr('reset')))
                print("Current largest army owner: %s" % (
                    "Player %d" % (game.players.index(
                        game.largest_army_owner) + 1) if game.largest_army_owner else "Nobody"))
                print("Player %d, you have these resources:" % (current_player_num + 1))
                for res, amount in current_player.resources.items():
                    print("    %s: %d" % (res, amount))
                print("and you have these development cards")
                for dev_card, amount in current_player.development_cards.items():
                    print("    %s: %d" % (dev_card, amount))
                print("Congratulations! Player %d wins!" % (current_player_num + 1))
                for i, player in enumerate(game.players):
                    # player_color = player_colors[i]
                    player_color = renderer.player_color_map[game.players[i]]
                    wins = winsPerPlayer[i]
                    print(stylize(f"Player {i + 1}: has won {wins}",
                                  fore_rgb(player_color[0], player_color[1], player_color[2])))
                print("Final board:")
                print(game.board)
                # print("Number of Turns: " + str(num_turns))
                num_games += 1
                for i in range(numAgents):
                    all_rewards[iteration][ep][i] = total_reward[i]
                    all_loss[iteration][ep] = np.sum(loss)
                    final_vps[iteration][ep][i] = game.get_victory_points(game.players[i])
                for i in range(len(agents)):
                    if isinstance(agents[i], ActorCritic) and i != current_player_num:
                        # print("Reward: " + str(all_rewards[iteration][ep][i]))
                        # agents[i].terminateEpisode(all_rewards[iteration][ep][i])
                        agents[i].terminateEpisode(0)
                    if isinstance(agents[i], ActorCritic) and i == current_player_num:
                        agents[i].terminateEpisode(100)
                # train the DQN agent
                for agent in agents:
                    if isinstance(agent, DQNAgent):
                        current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                         turn_reward, current_state, done=True)
                        print(choices)
                # saves the policy, rewards, and final VPs every save_every episodes
                if ep % save_every == 0:
                    save_progress(agents, all_rewards, final_vps, ep, iteration)
                break

            if type(agents[current_player_num]) is DQNAgent:
                current_state = process_dqn_step(agents[current_player_num], game, current_player_num,
                                                 turn_reward, current_state, done=True)

                # Train the agent
                if num_turns % 10 == 0:
                    test_loss = agents[0].train(ep, turn_reward)
                    if test_loss is not None:
                        loss.append(agents[0].train(ep, turn_reward))  # agent trains and returns the loss
            current_player_num = (current_player_num + 1) % len(game.players)


# the final policy is stored here
# actorCriticTrainedPolicy = []
# for i in range(numAgents):
#     if isinstance(agents[i], ActorCritic):
#         actorCriticTrainedPolicy.append(agents[i].theta)
# file = open("thetaResult.txt", "w")
# for i in range(len(actorCriticTrainedPolicy)):
#     print(str(actorCriticTrainedPolicy[i]))
#     file.write(str(actorCriticTrainedPolicy[i]) + "\n")
# file.close()
# print(num_games)

# agents[0].writer.close()

# Calculations for each agent
avg_rewards_agents = np.mean(all_rewards, axis=0)  # Shape: (num_episodes, numAgents)
std_rewards_agents = np.std(all_rewards, axis=0)

avg_loss = np.mean(all_loss, axis=0)
std_loss = np.mean(all_loss, axis=0)

avg_vps_agents = np.mean(final_vps, axis=0)
std_vps_agents = np.std(final_vps, axis=0)

# Plotting
episodes = np.arange(1, num_episodes + 1)

# Plot for average rewards
for agent in range(numAgents):
    avg_rewards = avg_rewards_agents[:, agent]
    std_rewards = std_rewards_agents[:, agent]
    upper_bound = avg_rewards + std_rewards
    lower_bound = avg_rewards - std_rewards

    # Plot the average rewards
    plt.plot(episodes, avg_rewards, label=f'Agent {agent + 1}')

    # Add the fill between for standard deviation
    plt.fill_between(episodes, lower_bound, upper_bound, alpha=0.5)  # 50% transparency

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward per Episode')
plt.legend()
plt.show()

# Your existing code for calculating the averages and standard deviations

for agent in range(numAgents):
    avg_vps = avg_vps_agents[:, agent]
    std_vps = std_vps_agents[:, agent]
    upper_bound = avg_vps + std_vps
    lower_bound = avg_vps - std_vps

    # Plot the average final victory points
    plt.plot(episodes, avg_vps, label=f'Agent {agent + 1}')

    # Add the fill between for standard deviation
    plt.fill_between(episodes, lower_bound, upper_bound, alpha=0.5)  # 50% transparency

plt.xlabel('Episode')
plt.ylabel('Average Final Victory Points')
plt.title('Average Final Victory Points per Episode')
plt.legend()
plt.show()

upper_bound = avg_loss + std_loss
lower_bound = avg_loss - std_loss
# Plot the average final victory points
plt.plot(range(len(avg_loss)), avg_loss, label=f'DQN Agent')

# Add the fill between for standard deviation
plt.fill_between(range(len(avg_loss)), lower_bound, upper_bound, alpha=0.5)  # 50% transparency

plt.xlabel('Number of Training Sessions')
plt.ylabel('Average Final Victory Points')
plt.title('Average Loss per Episode')
plt.legend()
plt.show()
