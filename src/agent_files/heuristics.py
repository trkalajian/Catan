# following are heuristics to code for when to choose certain actions.
from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import random

# build_settlement, if resources are available and a viable location exists, build
def build_settlement(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a settlement

    if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
        valid_settlement_coords = game.board.get_valid_settlement_coords(current_player, ensure_connected=True)
        if valid_settlement_coords:
            return [1, 1]


# places a settlement based on the available spot with the highest intersection value
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
    if current_player.has_resources(BuildingType.ROAD.get_required_resources()) and current_player.num_roads > 0:
        valid_road_coords = game.board.get_valid_road_coords(current_player, ensure_connected=True)
        if valid_road_coords:
            return [1, 3]

    return None  # Return None if no valid road placement is found or not enough resources


# choose where to place road, prioritizing the open road slot that leads to the intersection with the highest value
def choose_road_placement(game, current_player_num, is_start):
    current_player = game.players[current_player_num]
    valid_road_coords = game.board.get_valid_road_coords(current_player, ensure_connected=True)

    # If it's the start of the game, adjust valid_road_coords for settlements with no connected roads
    if is_start:
        unconnected_settlement_coords = []

        # Check each settlement of the current player
        for coords, intersection in game.board.intersections.items():
            if intersection.building is not None and intersection.building.owner == current_player and intersection.building.building_type == BuildingType.SETTLEMENT:
                connected_paths = game.board.get_paths_for_intersection_coords(coords)
                if not any(path.building is not None and path.building.owner == current_player for path in connected_paths):
                    unconnected_settlement_coords.append(coords)

        if unconnected_settlement_coords:
            # Adjust valid_road_coords to only include roads adjacent to unconnected settlements
            new_valid_road_coords = set()
            for settlement_coords in unconnected_settlement_coords:
                connected_intersections = game.board.get_intersection_connected_intersections(game.board.intersections[settlement_coords])
                for connected_intersection in connected_intersections:
                    path_coords = frozenset({settlement_coords, connected_intersection.coords})
                    if path_coords in game.board.paths:
                        new_valid_road_coords.add(path_coords)
            valid_road_coords = new_valid_road_coords.intersection(valid_road_coords)



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


def build_cities(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a city
    if current_player.has_resources(BuildingType.CITY.get_required_resources()):
        valid_city_coords = game.board.get_valid_city_coords(current_player)
        if valid_city_coords:
            return [1, 2]

    return None  # Return None if no valid city placement is found or not enough resources

def place_city(game, valid_city_coords):
    if not valid_city_coords:
        return None

    best_coords = None
    best_int_value = -1  # Initialize to a low value

    for coords in valid_city_coords:
        intersection_value = game.board.get_average_neighbor(coords)
        if intersection_value > best_int_value:
            best_coords = coords
            best_int_value = intersection_value

    return best_coords  # Return the best coordinates based on the highest average neighbor value

# You can now call this function within your game logic when it's appropriate to build a city


def build_dev_card(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a development card
    if current_player.has_resources(DevelopmentCard.get_required_resources()):
        # Check if there are development cards available
        if game.development_card_deck:
            return [1, 4]  # or any specific value you use to indicate the action to buy a development card

    return None  # Return False if conditions are not met


# Choose highest production tile not occupied by the current player but with some player on it and also select the player to steal from who has the most cards on the selected tile
def place_robber(game, current_player_num):
    current_player = game.players[current_player_num]
    highest_production_value = -1
    candidate_hexes = []

    for hex_coords, hex in game.board.hexes.items():
        # Get players on the hex
        players_on_hex = list(game.board.get_players_on_hex(hex_coords))

        # Skip the hex if no player has a building on it or if only the current player occupies it
        if not players_on_hex or (len(players_on_hex) == 1 and current_player in players_on_hex):
            continue

        # Calculate production value based on the token number
        token = hex.token_number
        if token in [2, 12]:
            production_value = 1
        elif token in [3, 11]:
            production_value = 2
        elif token in [4, 10]:
            production_value = 3
        elif token in [5, 9]:
            production_value = 4
        elif token in [6, 8]:
            production_value = 5
        else:
            production_value = 0  # Other numbers or no token

        # Track hexes with the highest production value
        if production_value > highest_production_value:
            highest_production_value = production_value
            candidate_hexes = [hex_coords]
        elif production_value == highest_production_value:
            candidate_hexes.append(hex_coords)

    # Randomly select one hex from the candidates
    target_hex_coords = random.choice(candidate_hexes) if candidate_hexes else None

    # Find the player with the most cards on the target hex
    if target_hex_coords:
        players_on_target_hex = list(game.board.get_players_on_hex(target_hex_coords))
        target_player = max(players_on_target_hex, key=lambda p: sum(p.resources.values()), default=None)

        return target_hex_coords, target_player

    return None, None  # Return None if no suitable hex and player are found

def play_knight(game, current_player_num):
    current_player = game.players[current_player_num]

    # Check if the player has a Knight development card
    if current_player.development_cards.get(DevelopmentCard.KNIGHT, 0) > 0:
        return [3, None]

    return None  # Return None if the player does not have a Knight card

# checks if  player has >=8 cards and 4 or 3 of a kind of one resource (3 if player has a harbor) and if so trades in
def should_trade_with_bank_or_port(game, current_player_num):
    current_player = game.players[current_player_num]
    resources = current_player.resources

    # Check if total resources are greater than or equal to 7
    if sum(resources.values()) >= 8:
        # Check for 4 of a kind without port access or 3 of a kind with port access
        resource_threshold = 3 if current_player.connected_harbors else 4

        for resource_count in resources.values():
            if resource_count >= resource_threshold:
                return [2, None]

    return None  # Return None if conditions for trading are not met

def calculate_production_value(token):
    if token in [2, 12]:
        return 1
    elif token in [3, 11]:
        return 2
    elif token in [4, 10]:
        return 3
    elif token in [5, 9]:
        return 4
    elif token in [6, 8]:
        return 5
    else:
        return 0

# looks at the current players owned intersections and counts up the production value for each resource tile occupied
def calculate_resource_production_totals(game, current_player):
    current_player = current_player
    production_totals = {'LUMBER': 0, 'WOOL': 0, 'GRAIN': 0, 'BRICK': 0, 'ORE': 0}

    for intersection_coords, intersection in game.board.intersections.items():
        if intersection.building and intersection.building.owner == current_player:
            adjacent_hex_coords = game.board.get_hexes_connected_to_intersection(intersection_coords)

            for hex_coord in adjacent_hex_coords:
                hex_tile = game.board.hexes.get(hex_coord)
                if hex_tile and hex_tile.hex_type.get_resource() is not None:
                    resource = hex_tile.hex_type.get_resource().name
                    yield_value = calculate_production_value(hex_tile.token_number)

                    # Add yield value to the total production of the corresponding resource
                    if resource in production_totals:
                        production_totals[resource] += yield_value

    return production_totals

# with the trade action selected choose the best one by taking the resource the player has the lowest production
# capacity for. If that resource is being traded choose second most
def choose_best_trade(game, current_player_num, possible_trades):
    current_player = game.players[current_player_num]
    resource_production_totals = calculate_resource_production_totals(game, current_player)

    # Sort resources by production total
    sorted_resources = sorted(resource_production_totals.items(), key=lambda x: x[1])

    # Identify the resources with the lowest and second lowest production totals
    resource_with_lowest_total = sorted_resources[0][0] if sorted_resources else None
    second_lowest_resource = sorted_resources[1][0] if len(sorted_resources) > 1 else None

    best_trade = None
    lowest_cards_discarded = float('inf')

    for trade in possible_trades:
        # Get the resources being offered and returned in the trade
        offered_resource = list(trade.keys())[0]
        returned_resource = list(trade.keys())[1]

        # Determine the appropriate resource to compare
        compare_resource = resource_with_lowest_total
        if offered_resource.name.lower() == resource_with_lowest_total.lower() and second_lowest_resource:
            compare_resource = second_lowest_resource

        # Skip this trade if it doesn't return the appropriate resource
        if returned_resource.name.lower() != compare_resource.lower():
            continue

        # Count the number of cards to be discarded
        cards_discarded = -sum(value for value in trade.values() if value < 0)

        # Select the trade that discards the fewest cards
        if cards_discarded < lowest_cards_discarded:
            best_trade = trade
            lowest_cards_discarded = cards_discarded

    return best_trade

def heuristic_policy(game, current_player_num):
    # Try playing a knight card
    knight_action = play_knight(game, current_player_num)
    if knight_action is not None:
        return knight_action

    # Try trading with bank or port
    trade_action = should_trade_with_bank_or_port(game, current_player_num)
    if trade_action is not None:
        return trade_action

    # Try building a city
    city_action = build_cities(game, current_player_num)
    if city_action is not None:
        return city_action

    # Try building a settlement
    settle_action = build_settlement(game, current_player_num)
    if settle_action is not None:
        return settle_action

    # Try building roads
    road_action = build_roads(game, current_player_num)
    if road_action is not None:
        return road_action

    # Try buying a development card
    dev_card_action = build_dev_card(game, current_player_num)
    if dev_card_action is not None:
        return dev_card_action

    # Default action
    return [4, None]

# Include the previously defined functions here (play_knight, should_trade_with_bank_or_port, build_cities, build_roads, build_dev_card)




