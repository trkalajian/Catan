from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
from methods import choose_path, choose_intersection, choose_resource, move_robber, count_cards, resource_check, choose_hex, get_coord_sort_by_xy
from agent_files.heuristics import heuristic_policy
from agent_files.agent import HeuristicAgent
from agent_files.heuristics import build_settlement, place_settlement, heuristic_policy, choose_road_placement, place_robber, choose_best_trade, place_city
import actorcritic
import random
import sys
import numpy as np
from actorcritic import ActorCritic

#set number of players here
numAgents = 2
agents = []
winsPerPlayer = np.zeros(numAgents)
#reward function for things other than winning

    

# Function to create a new board and game
def create_new_game():
    board = BeginnerBoard()
    # BoardRenderer.player_color_map = {}
    # BoardRenderer._unused_player_colors = BoardRenderer.DEFAULT_PLAYER_COLORS
    game = Game(board, numAgents)
    renderer = BoardRenderer(game.board,{})
    return game, renderer


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
for i in range(numAgents):
    # change some of these to actor-critics when appropriate

    if i >= 1: 
        agent = heuristic_agent_maker()

        agents.append(agent)
    else:
        #creates an actor critic
        agents.append(actorcritic.ActorCritic(place_settlement_func=place_settlement,
                                                 place_road_func=choose_road_placement,
                                                 place_robber_func = place_robber,
                                                 choose_best_trade=choose_best_trade,
                                                 place_city_func=place_city))


# Main game loop
current_player_num = 0
is_start = False
num_turns = 0
num_games = 0
max_turns = 1000  # Number of turns to play
max_games = 1000


while num_games < max_games:
    print("Game Number: " + str(num_games))

    # Create a new board and game for each iteration to reset the board
    game, renderer = create_new_game()
    # Creating players and setting the order
    player_order = list(range(len(game.players)))
    random.shuffle(player_order)

    is_start = False
    for i in player_order + list(reversed(player_order)):
        is_start = True
        current_player = game.players[i]
        if isinstance(agents[i], ActorCritic):
            agents[i].initializeEpisode(game, current_player)
        # coords = choose_intersection(game.board.get_valid_settlement_coords(current_player, ensure_connected=False),
        #                              "Where do you want to build your settlement? ")
        agent_coords = agents[i].place_settlement(game, i, is_start)
        game.build_settlement(player=current_player, coords=agent_coords, cost_resources=False, ensure_connected=False)
        current_player.add_resources(game.board.get_hex_resources_for_intersection(agent_coords))
        # road_options = game.board.get_valid_road_coords(current_player, connected_intersection=agent_coords)
        road_coords = agents[i].place_road(game, i)
        game.build_road(player=current_player, path_coords=road_coords, cost_resources=False)

    print(game.board)
    current_player_num = 0
    num_turns = 0

    is_start = False
    
    while num_turns < max_turns:
        current_player = game.players[current_player_num]
        # Roll the dice
        dice = random.randint(1, 6) + random.randint(1, 6)
        if dice == 7:
            card_totals = count_cards(game)
            resource_check(card_totals, game)
            hex_coords, player_stolen = agents[current_player_num].place_robber(game, current_player_num)
            move_robber(current_player, game, hex_coords, player_stolen)
            pass
        else:
            game.add_yield_for_roll(dice)
        choice = [0,0]
        while choice[0] != 4:
            while True:
                try:
                    choice = agents[current_player_num].policy(game, current_player_num)
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
                    coords = agents[current_player_num].place_settlement(game, current_player_num, is_start)
                    game.build_settlement(current_player, coords)
                elif building_choice == 2:
                    if not current_player.has_resources(BuildingType.CITY.get_required_resources()):
                        continue
                    valid_coords = game.board.get_valid_city_coords(current_player)
                    # coords = choose_intersection(valid_coords, "Where do you want to build a city?  ", game, renderer)
                    coords = agents[current_player_num].place_city_func(game, valid_coords)
                    game.upgrade_settlement_to_city(current_player, coords)
                elif building_choice == 3:
                    if not current_player.has_resources(BuildingType.ROAD.get_required_resources()):
                        continue
                    valid_coords = game.board.get_valid_road_coords(current_player)
                    if not valid_coords:
                        continue
                    path_coords = agents[current_player_num].place_road(game, current_player_num)
                    game.build_road(current_player, path_coords)
                elif building_choice == 4:
                    if not current_player.has_resources(DevelopmentCard.get_required_resources()):
                        continue
                    dev_card = game.build_development_card(current_player)
                elif building_choice == 5:
                    break

            elif choice[0] == 2:
                possible_trades = list(current_player.get_possible_trades())
                # trade_choice = int(input('->  '))
                trade = agents[current_player_num].choose_trade_func(game, current_player_num, possible_trades)
                if trade == None:
                    print('woa')
                current_player.add_resources(trade)
            elif choice[0] == 3:
                dev_cards = [card for card, amount in current_player.development_cards.items() if
                             amount > 0 and card is not DevelopmentCard.VICTORY_POINT]
                card_to_play = DevelopmentCard.KNIGHT
                game.play_development_card(current_player, card_to_play)
                if card_to_play is DevelopmentCard.KNIGHT:
                    hex_coords, player_stolen = agents[current_player_num].place_robber(game, current_player_num)
                    move_robber(current_player, game, hex_coords, player_stolen)
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
            if isinstance(agents[current_player_num], ActorCritic):
                #This needs to pass in a reward that includes winning the game, put in 100 for now
                agents[current_player_num].terminateEpisode(100)
            for i in range(len(agents)):
                if i != current_player_num and isinstance(agents[i], ActorCritic):
                    #rewards players who didnt win based on their VP number
                    agents[i].terminateEpisode(game.get_victory_points(game.players[i]))
                    

                
                
            print("Current Victory point standings:")
            for i in range(len(game.players)):
                print("Player %d: %d VP" % (i + 1, game.get_victory_points(game.players[i])))
            print("Current longest road owner: %s" % (
                "Player %d" % (game.players.index(
                    game.longest_road_owner) + 1) if game.longest_road_owner else "Nobody"))
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
            for i in range(len(game.players)):
                print("Player %d has won %d times " % ((i +1), winsPerPlayer[i]))
            print("Final board:")
            print(game.board)
            print("Number of Turns: " + str(num_turns))
            num_games += 1
            break

        current_player_num = (current_player_num + 1) % len(game.players)
        num_turns += 1
    if num_turns == max_turns:
        print("Number of Turns: " + str(num_turns))
        for i in range(len(game.players)):
            print("Player %d has won %d times " % ((game.players[i] + 1), winsPerPlayer[game.players[i]]))

        num_games += 1
        for i in range(len(agents)):
            agents[i].terminateEpisode(game.get_victory_points(game.players[i]))
            
#the final policy is stored here
actorCriticTrainedPolicy = []
for i in agents[i]:
    if isinstance(agents[i], ActorCritic):
        actorCriticTrainedPolicy.append(agents[i].theta)
print(num_games)