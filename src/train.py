from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import string
import random
import sys
import math
from play import choose_path, choose_hex, choose_resource, move_robber, count_cards, resource_check
import random

# Player order for settlements and roads building
player_order = list(range(len(game.players)))
is_start = False
for i in player_order + list(reversed(player_order)):
    is_start = True
    current_player = game.players[i]
    coords = choose_intersection(game.board.get_valid_settlement_coords(current_player, ensure_connected=False),
                                 "Where do you want to build your settlement? ")
    game.build_settlement(player=current_player, coords=coords, cost_resources=False, ensure_connected=False)
    current_player.add_resources(game.board.get_hex_resources_for_intersection(coords))
    road_options = game.board.get_valid_road_coords(current_player, connected_intersection=coords)
    road_coords = choose_path(road_options, "Where do you want to build your road to? ")
    game.build_road(player=current_player, path_coords=road_coords, cost_resources=False)

# Main game loop
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
            sys.exit(0)

    current_player_num = (current_player_num + 1) % len(game.players)