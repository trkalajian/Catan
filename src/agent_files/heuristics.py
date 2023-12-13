# following are heuristics to code for when to choose certain actions.
from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import random

# build_settlement, if resources are available and a viable location exists, build
def build_settlement(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a settlement

    if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
        valid_settlement_coords = game.board.get_valid_settlement_coords(current_player, ensure_connected=False)
        if valid_settlement_coords:
            return [1, 1]


def place_settlement(game, current_player_num, is_start):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a settlement
    if is_start:
        valid_settlement_coords = game.board.get_valid_settlement_coords(current_player, ensure_connected=False)
    else:
        valid_settlement_coords = game.board.get_valid_settlement_coords(current_player, ensure_connected=True)

    if valid_settlement_coords:
        best_coords = None
        best_int_value = -1  # Initialize to a low value

        for coords in valid_settlement_coords:
            intersection_value = game.board.get_average_neighbor(coords)
            total_value = 0
            num_neighbors = 0
            if intersection_value > best_int_value:
                best_coords = coords
                best_int_value = intersection_value

        if best_coords:
            return best_coords

    return None  # Return None if no valid settlement coordinates are found

# function to output the selections for building a road
def build_roads(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a road
    if current_player.has_resources(BuildingType.ROAD.get_required_resources()):
        valid_road_coords = game.board.get_valid_road_coords(current_player, ensure_connected=True)
        if valid_road_coords:
            return [1, 3]

    return None  # Return None if no valid road placement is found or not enough resources


# choose where to place road
def choose_road_placement(game, current_player_num):
    current_player = game.players[current_player_num]
    valid_road_coords = game.board.get_valid_road_coords(current_player, ensure_connected=True)

    if valid_road_coords:
        best_road_coords = None
        highest_neighbor_value = -1  # Starting with a low value

        for path_coords in valid_road_coords:
            for coords in path_coords:
                # Check if the road leads to a new intersection
                if game.board.intersections[coords].building is None or game.board.intersections[coords].building.owner == current_player:
                    intersection_value = game.board.get_average_neighbor(coords)

                    # Select the road leading to the intersection with the highest value
                    if intersection_value > highest_neighbor_value:
                        best_road_coords = path_coords
                        highest_neighbor_value = intersection_value

        return best_road_coords

    return None  # Return None if no valid road placement is found


    return None  # Return None if no valid road placement is found


def build_cities(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a city
    if current_player.has_resources(BuildingType.CITY.get_required_resources()):
        valid_city_coords = game.board.get_valid_city_coords(current_player)
        if valid_city_coords:
            return [1, 2]

    return None  # Return None if no valid city placement is found or not enough resources


def build_dev_card(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a development card
    if current_player.has_resources(DevelopmentCard.get_required_resources()):
        # Check if there are development cards available
        if game.development_card_deck:
            return [1, 4]  # or any specific value you use to indicate the action to buy a development card

    return False  # Return False if conditions are not met


# def place robber:
    # Choose highest production tile of the current point leader that you also don't occupy

# def play dev card:
    # if dev card is in hand play it
    # knight: def place robber

# def trade w/ bank or port
    # if port is available trade with that of course
    # don't trade in until your resources are >=7 and have 3/4 of a kind

# def choose resource (used if monopoly is called or for trading)
    # choose resource with lowest resource production for player

# def resource_production (add to _player.py maybe)
    # +1 for (2,12), +2 for (3, 11), +3 for (4,10), +4 for (5,9), +5 for (6,8) for each resource you have

def heuristic_policy(game, current_player_num, is_start):
    # hierarchy of selecting actions

    build_settlement(game, current_player_num)



