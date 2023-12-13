from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
from methods import choose_path, choose_intersection, choose_resource, move_robber, count_cards, resource_check, choose_hex, get_coord_sort_by_xy
from agent_files.heuristics import heuristic_policy
from agent_files.agent import HeuristicAgent
from agent_files.heuristics import build_settlement, place_settlement, heuristic_policy, choose_road_placement
import actorcritic
import random
import sys


#initialize agents before starting games
numAgents = 2
agents = []

for i in range(numAgents):
    #change some of these to actor-critics when appropriate
        
    #agent = heuristic_agent_maker()
    #agents.append(agent)
    agents.append(actorcritic.ActorCritic(build_settlement_func=build_settlement,
        place_settlement_func=place_settlement,
        place_road_func=choose_road_placement))
# Function to create a new board and game
def create_new_game():
    board = BeginnerBoard()
    game = Game(board, numAgents)
    renderer = BoardRenderer(game.board)
    return game


def heuristic_agent_maker():
    heur_policy = heuristic_policy
    return HeuristicAgent(
        policy=heur_policy,
        build_settlement_func=build_settlement,
        place_settlement_func=place_settlement,
        place_road_func=choose_road_placement
    )


# Main game loop
current_player_num = 0
is_start = False
num_turns = 0
max_turns = 10000  # Number of turns to play
while num_turns < max_turns:
    # Create a new board and game for each iteration to reset the board

    game = create_new_game()
    renderer = BoardRenderer(game.board)

    # Creating players and setting the order
    player_order = list(range(len(game.players)))
    
 


    is_start = False
    for i in player_order + list(reversed(player_order)):
        is_start = True
        current_player = game.players[i]
        # coords = choose_intersection(game.board.get_valid_settlement_coords(current_player, ensure_connected=False),
        #                              "Where do you want to build your settlement? ")
        agents[i].initializeEpisode(game, game.players[i])
        agent_coords = agents[i].place_settlement(game, i, is_start)
        game.build_settlement(player=current_player, coords=agent_coords, cost_resources=False, ensure_connected=False)
        current_player.add_resources(game.board.get_hex_resources_for_intersection(agent_coords))
        # road_options = game.board.get_valid_road_coords(current_player, connected_intersection=agent_coords)
        road_coords = agents[i].place_road(game, i)
        game.build_road(player=current_player, path_coords=road_coords, cost_resources=False)

    print(game.board)
    current_player_num = 0
    is_start = False
    while True:
        current_player = game.players[current_player_num]
        # Roll the dice
        dice = random.randint(1, 6) + random.randint(1, 6)
        if dice == 7:
            card_totals = count_cards(game)
            resource_check(card_totals, game)
            move_robber(current_player)
            pass
        else:
            game.add_yield_for_roll(dice)
        choice = 0
        while choice != 4:
            while True:
                try:
                    choice = agents[current_player_num].policy(game)
                    if 1 <= choice <= 4:
                        break  # Valid choice, exit the loop
                except ValueError:
                    print("Invalid input. Please enter a number.")
            if choice == 1:
                building_choice = int(input('->  '))
                if not 1 <= building_choice <= 5:
                    continue
                if building_choice == 1:
                    if not current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
                        continue
                    valid_coords = game.board.get_valid_settlement_coords(current_player)
                    if not valid_coords:
                        continue
                    coords = choose_intersection(valid_coords, "Where do you want to build a settlement?  ")
                    game.build_settlement(current_player, coords)
                elif building_choice == 2:
                    if not current_player.has_resources(BuildingType.CITY.get_required_resources()):
                        continue
                    valid_coords = game.board.get_valid_city_coords(current_player)
                    coords = choose_intersection(valid_coords, "Where do you want to build a city?  ")
                    game.upgrade_settlement_to_city(current_player, coords)
                elif building_choice == 3:
                    if not current_player.has_resources(BuildingType.ROAD.get_required_resources()):
                        continue
                    valid_coords = game.board.get_valid_road_coords(current_player)
                    if not valid_coords:
                        continue
                    path_coords = choose_path(valid_coords, "Where do you want to build a road?")
                    game.build_road(current_player, path_coords)
                elif building_choice == 4:
                    if not current_player.has_resources(DevelopmentCard.get_required_resources()):
                        continue
                    dev_card = game.build_development_card(current_player)
                elif building_choice == 5:
                    break

            elif choice == 2:
                possible_trades = list(current_player.get_possible_trades())
                trade_choice = int(input('->  '))
                trade = possible_trades[trade_choice]
                current_player.add_resources(trade)
            elif choice == 3:
                dev_cards = [card for card, amount in current_player.development_cards.items() if
                             amount > 0 and card is not DevelopmentCard.VICTORY_POINT]
                card_to_play = dev_cards[int(input('->  '))]
                game.play_development_card(current_player, card_to_play)
                if card_to_play is DevelopmentCard.KNIGHT:
                    move_robber(current_player)
                elif card_to_play is DevelopmentCard.YEAR_OF_PLENTY:
                    for _ in range(2):
                        resource = choose_resource("What resource do you want to receive?")
                        current_player.add_resources({resource: 1})
                elif card_to_play is DevelopmentCard.ROAD_BUILDING:
                    for _ in range(2):
                        valid_path_coords = game.board.get_valid_road_coords(current_player)
                        path_coords = choose_path(valid_path_coords, "Choose where to build a road: ")
                        game.build_road(current_player, path_coords, cost_resources=False)
                elif card_to_play is DevelopmentCard.MONOPOLY:
                    resource = choose_resource("What resource do you want to take?")
                    for i in range(len(game.players)):
                        player = game.players[i]
                        if player is not current_player:
                            amount = player.resources[resource]
                            player.remove_resources({resource: amount})
                            current_player.add_resources({resource: amount})
            if game.get_victory_points(current_player) >= 10:
                for playerNumber in range(len(game.players)):
                    if playerNumber == current_player_num:
                        finalReward = 100
                    else:
                        finalReward = 0
                    agent[playerNumber].terminateEpisode(finalReward)
                        
                
                #we should start another game instead of exiting to train actor critic since it learns by taking actions
                sys.exit(0)

        current_player_num = (current_player_num + 1) % len(game.players)
        num_turns += 1
