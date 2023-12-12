# following are heuristics to code for when to choose certain actions.
from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType

# build_settlement, if resources are available and a viable location exists, build
def build_settlement_heuristic(game, current_player_num, is_start):
    current_player = game.players[current_player_num]

    # Check if the player has enough resources to build a settlement
    if current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
        valid_settlement_coords = game.board.get_valid_settlement_coords(current_player, ensure_connected=False)
        if valid_settlement_coords:
            settlement_coords = valid_settlement_coords[0]  # Choose the first valid location (you can modify this)
            game.build_settlement(current_player, settlement_coords, cost_resources=False, ensure_connected=False)
            current_player.add_resources(game.board.get_hex_resources_for_intersection(settlement_coords))
            # Print a message indicating that a settlement is built using the heuristic
            print(f"Player {current_player_num + 1} built a settlement using the heuristic at {settlement_coords}")

# def choose where to place settlement
    # max average value each available intersection

# def choose to build road
    # if resources are available and longest_road == no or viable settlement spots == 0

# def choose where to place road
    # if road connects to another road yes
    # else randomly choose among viable spots

# def choose to build dev card:
    # if resources are available and dev cards are available, buy

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

#


