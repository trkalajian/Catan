import numpy as np

from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import string
import math
import random

def get_coord_sort_by_xy(c, renderer):
    x, y = renderer.get_coords_as_xy(c)
    return 1000 * x + y

# Letters to label the positions on the board
label_letters = string.ascii_lowercase + string.ascii_uppercase + "123456789"

# Function to let the player choose an intersection to build
def choose_intersection(intersection_coords, prompt, game, renderer):
    # Label all the letters on the board
    intersection_list = [game.board.intersections[i] for i in intersection_coords]
    intersection_list.sort(key=lambda i: get_coord_sort_by_xy(i.coords))
    intersection_labels = {intersection_list[i]: label_letters[i] for i in range(len(intersection_list))}
    renderer.render_board(intersection_labels=intersection_labels)

    while True:
        # Prompt the user
        letter = input(prompt)
        letter_to_intersection = {v: k for k, v in intersection_labels.items()}

        if letter in letter_to_intersection:
            intersection = letter_to_intersection[letter]
            return intersection.coords
        else:
            print(f"The letter '{letter}' is not available. Choose another location.")

# Function to let the player choose a path to build
def choose_path(path_coords, game, renderer):
    # Label all the paths with a letter
    path_list = [game.board.paths[i] for i in path_coords]
    path_labels = {path_list[i]: label_letters[i] for i in range(len(path_coords))}
    renderer.render_board(path_labels=path_labels)

    while True:
        # Ask the user for a letter
        letter = input()[0]
        # Get the path from the letter entered by the user
        letter_to_path = {v: k for k, v in path_labels.items()}

        if letter in letter_to_path:
            return letter_to_path[letter].path_coords
        else:
            print(f"The letter '{letter}' is not available. Choose another path.")


# Function to let the player choose a hex
def choose_hex(hex_coords, prompt, game, renderer):
    # Label all the hexes with a letter
    hex_list = [game.board.hexes[i] for i in hex_coords]
    hex_list.sort(key=lambda h: get_coord_sort_by_xy(h.coords))
    hex_labels = {hex_list[i]: label_letters[i] for i in range(len(hex_list))}
    renderer.render_board(hex_labels=hex_labels)
    while True:
        letter = input(prompt)
        letter_to_hex = {v: k for k, v in hex_labels.items()}

        if letter in letter_to_hex:
            return letter_to_hex[letter].coords
        else:
            print(f"The letter '{letter}' is not available. Choose another path.")

# Function to let the player choose a resource
def choose_resource(prompt):
    print(prompt)
    resources = [res for res in Resource]
    for i in range(len(resources)):
        print("%d: %s" % (i, resources[i]))
    resource_choice = int(input('->  '))
    return resources[resource_choice]

# Function to let the player move the robber and steal a card
def move_robber(player, game, hex_coords, player_stolen):
    # hex_coords = choose_hex([c for c in game.board.hexes if c != game.board.robber],
    #                         "Where do you want to move the robber? ")
    game.board.robber = hex_coords
    # Choose a player to steal a card from
    # potential_players = list(game.board.get_players_on_hex(hex_coords))
    # if not potential_players:
    #     return
    # for p in potential_players:
    #     i = game.players.index(p)
    #     print("%d: Player %d" % (i + 1, i + 1))
    # p = int(input('->  ')) - 1
    # If they try and steal from another player they lose their chance to steal
    # to_steal_from = game.players[p] if game.players[p] in potential_players else None
    to_steal_from = player_stolen
    if to_steal_from:
        resource = to_steal_from.get_random_resource()
        if resource is not None:
            player.add_resources({resource: 1})
            to_steal_from.remove_resources({resource: 1})
        # print("Stole 1 %s for player %d" % (resource, p + 1))

def count_cards(game):
    card_totals = []
    for player in game.players:
        total = sum(player.resources.values())
        card_totals.append(total)
    return card_totals

# takes any player over 7 cards and performs the discard action randomly
def   resource_check(card_totals, game):
    discard_rewards = np.zeros(len(card_totals))  # Initialize rewards array

    for i, total in enumerate(card_totals):
        if total > 7:
            player = game.players[i]
            print(f"Player {i + 1} has more than 7 cards ({total} cards). Discarding Resources.")

            # Calculate the number of cards to discard (half, rounding up)
            cards_to_discard = math.ceil(total / 2)

            # Get a list of all resources the player has
            all_resources = list(player.resources.keys())

            # Randomly select cards to discard
            for _ in range(cards_to_discard):
                if not all_resources:
                    break  # No more resources to discard
                while True:
                    resource_to_discard = random.choice(all_resources)
                    if player.resources[resource_to_discard] > 0:
                        break
                player.resources[resource_to_discard] -= 1

            # Update the rewards array only if cards are discarded
            if cards_to_discard > 0:
                discard_rewards[i] = -1 * cards_to_discard

            print(f"Player {i + 1} discarded {cards_to_discard}")
        else:
            # Reward for not discarding any cards
            discard_rewards[i] = 0

    return discard_rewards
