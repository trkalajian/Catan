from pycatan import Game, DevelopmentCard, Resource
from pycatan.board import BeginnerBoard, BoardRenderer, BuildingType
import string
import random
import sys
import math
import random
from features import Features

# Create a Catan game with a beginner boards
game = Game(BeginnerBoard(), 2)
renderer = BoardRenderer(game.board)


newfeature = Features(game, game.players[0])


# Function to sort coordinates based on (x, y)
def get_coord_sort_by_xy(c):
    x, y = renderer.get_coords_as_xy(c)
    return 1000 * x + y

# Letters to label the positions on the board
label_letters = string.ascii_lowercase + string.ascii_uppercase + "123456789"

# Function to let the player choose an intersection to build
def choose_intersection(intersection_coords, prompt):
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
def choose_path(path_coords, prompt):
    # Label all the paths with a letter
    path_list = [game.board.paths[i] for i in path_coords]
    path_labels = {path_list[i]: label_letters[i] for i in range(len(path_coords))}
    renderer.render_board(path_labels=path_labels)

    while True:
        # Ask the user for a letter
        letter = input(prompt)[0]
        # Get the path from the letter entered by the user
        letter_to_path = {v: k for k, v in path_labels.items()}

        if letter in letter_to_path:
            return letter_to_path[letter].path_coords
        else:
            print(f"The letter '{letter}' is not available. Choose another path.")


# Function to let the player choose a hex
def choose_hex(hex_coords, prompt):
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
def move_robber(player):
    # Don't let the player move the robber back onto the same hex
    hex_coords = choose_hex([c for c in game.board.hexes if c != game.board.robber],
                            "Where do you want to move the robber? ")
    game.board.robber = hex_coords
    # Choose a player to steal a card from
    potential_players = list(game.board.get_players_on_hex(hex_coords))
    if not potential_players:
        print("No potential players to steal from.")
        return
    print("Choose who you want to steal from:")
    for p in potential_players:
        i = game.players.index(p)
        print("%d: Player %d" % (i + 1, i + 1))
    p = int(input('->  ')) - 1
    # If they try and steal from another player they lose their chance to steal
    to_steal_from = game.players[p] if game.players[p] in potential_players else None
    if to_steal_from:
        resource = to_steal_from.get_random_resource()
        player.add_resources({resource: 1})
        to_steal_from.remove_resources({resource: 1})
        print("Stole 1 %s for player %d" % (resource, p + 1))

def count_cards(game):
    card_totals = []
    for player in game.players:
        total = sum(player.resources.values())
        card_totals.append(total)
    return card_totals

# takes any player over 7 cards and performs the discard action randomly
def resource_check(card_totals, game):
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

            # print(f"Player {i + 1} discarded {cards_to_discard}. Remaining cards: {player.resources}")
            print(f"Player {i + 1} discarded {cards_to_discard}")

# Player order for settlements and roads building
player_order = list(range(len(game.players)))
for i in player_order + list(reversed(player_order)):
    current_player = game.players[i]
    print("Player %d, it is your turn!" % (i + 1))
    coords = choose_intersection(game.board.get_valid_settlement_coords(current_player, ensure_connected=False),
                                 "Where do you want to build your settlement? ")
    game.build_settlement(player=current_player, coords=coords, cost_resources=False, ensure_connected=False)
    current_player.add_resources(game.board.get_hex_resources_for_intersection(coords))
    # Print the road options
    road_options = game.board.get_valid_road_coords(current_player, connected_intersection=coords)
    road_coords = choose_path(road_options, "Where do you want to build your road to? ")
    game.build_road(player=current_player, path_coords=road_coords, cost_resources=False)

# Counts how many resource cards a player has and returns an array with those counts

# Main game loop
current_player_num = 0
while True:
    current_player = game.players[current_player_num]
    print("Player %d, it is your turn now" % (current_player_num + 1))
    # Roll the dice
    dice = random.randint(1, 6) + random.randint(1, 6)
    # dice = 7 # for testing robber
    print("Player %d rolled a %d" % (current_player_num + 1, dice))
    if dice == 7:
        card_totals = count_cards(game)
        resource_check(card_totals, game)
        move_robber(current_player)
        pass
    else:
        game.add_yield_for_roll(dice)
    choice = 0
    while choice != 4:
        print(game.board)
        print("Current Victory point standings:")
        for i in range(len(game.players)):
            print("Player %d: %d VP" % (i + 1, game.get_victory_points(game.players[i])))
        print("Current longest road owner: %s" % (
            "Player %d" % (game.players.index(game.longest_road_owner) + 1) if game.longest_road_owner else "Nobody"))
        print("Current largest army owner: %s" % (
            "Player %d" % (game.players.index(game.largest_army_owner) + 1) if game.largest_army_owner else "Nobody"))
        print("Player %d, you have these resources:" % (current_player_num + 1))
        for res, amount in current_player.resources.items():
            print("    %s: %d" % (res, amount))
        print("and you have these development cards")
        for dev_card, amount in current_player.development_cards.items():
            print("    %s: %d" % (dev_card, amount))
        print("Choose what to do:")
        print("1 - Build something")
        print("2 - Trade")
        print("3 - Play a dev card")
        print("4 - Next player's turn")
        while True:
            try:
                choice = int(input("->  "))
                if 1 <= choice <= 4:
                    break  # Valid choice, exit the loop
            except ValueError:
                print("Invalid input. Please enter a number.")
        if choice == 1:
            print("What do you want to build? ")
            print("1 - Settlement")
            print("2 - City")
            print("3 - Road")
            print("4 - Development Card")
            print("5 - Back")
            while True:
                building_choice = int(input('->  '))
                if not 1<= building_choice <=5:
                    print("Invalid input. Please enter a new number.")
                else:
                    break
            if building_choice == 1:
                if not current_player.has_resources(BuildingType.SETTLEMENT.get_required_resources()):
                    print("You don't have enough resources to build a settlement")
                    continue
                # Get the valid settlement coords
                valid_coords = game.board.get_valid_settlement_coords(current_player)
                if not valid_coords:
                    print("There are no valid positions to build a settlement")
                    continue
                # Get the player to choose one
                coords = choose_intersection(valid_coords, "Where do you want to build a settlement?  ")
                game.build_settlement(current_player, coords)
            elif building_choice == 2:
                # Check the player has enough resources
                if not current_player.has_resources(BuildingType.CITY.get_required_resources()):
                    print("You don't have enough resources to build a city")
                    continue
                # Get the valid coords to build a city at
                valid_coords = game.board.get_valid_city_coords(current_player)
                # Have the player choose one
                coords = choose_intersection(valid_coords, "Where do you want to build a city?  ")
                # Build the city
                game.upgrade_settlement_to_city(current_player, coords)
            elif building_choice == 3:
                # Check the player has enough resources
                if not current_player.has_resources(BuildingType.ROAD.get_required_resources()):
                    print("You don't have enough resources to build a road")
                    continue
                # Get the valid road coordinates
                valid_coords = game.board.get_valid_road_coords(current_player)
                # If there are none
                if not valid_coords:
                    print("There are no valid places to build a road")
                    continue
                # Have the player choose one
                path_coords = choose_path(valid_coords, "Where do you want to build a road?")
                game.build_road(current_player, path_coords)
            elif building_choice == 4:
                # Check the player has the resources to build a development card
                if not current_player.has_resources(DevelopmentCard.get_required_resources()):
                    print("You do not have the resources to build a development card")
                    continue
                # Build a card and tell the player what they build
                dev_card = game.build_development_card(current_player)
                print("You built a %s card" % dev_card)
            elif building_choice == 5:
                break

        elif choice == 2:
            possible_trades = list(current_player.get_possible_trades())
            print("Choose a trade: ")
            for i in range(len(possible_trades)):
                print("%d:" % i)
                for res, amount in possible_trades[i].items():
                    print("    %s: %d" % (res, amount))
            trade_choice = int(input('->  '))
            trade = possible_trades[trade_choice]
            current_player.add_resources(trade)
        elif choice == 3:
            # Choose a development card
            print("What card do you want to play?")
            dev_cards = [card for card, amount in current_player.development_cards.items() if
                         amount > 0 and card is not DevelopmentCard.VICTORY_POINT]
            for i in range(len(dev_cards)):
                print("%d: %s" % (i, dev_cards[i]))
            card_to_play = dev_cards[int(input('->  '))]

            # Doesn't actually do anything but remove the card from the player's hand and recalculate largest army
            game.play_development_card(current_player, card_to_play)
            if card_to_play is DevelopmentCard.KNIGHT:
                move_robber(current_player)
            elif card_to_play is DevelopmentCard.YEAR_OF_PLENTY:
                # Have the player choose 2 resources to receive
                for _ in range(2):
                    resource = choose_resource("What resource do you want to receive?")
                    # Add that resource to the player's hand
                    current_player.add_resources({resource: 1})
            elif card_to_play is DevelopmentCard.ROAD_BUILDING:
                # Allow the player to build 2 roads
                for _ in range(2):
                    valid_path_coords = game.board.get_valid_road_coords(current_player)
                    path_coords = choose_path(valid_path_coords, "Choose where to build a road: ")
                    game.build_road(current_player, path_coords, cost_resources=False)
            elif card_to_play is DevelopmentCard.MONOPOLY:
                # Choose a resource
                resource = choose_resource("What resource do you want to take?")
                # Remove that resource from everyone else's hands and add it to the current player's hand
                for i in range(len(game.players)):
                    player = game.players[i]
                    if player is not current_player:
                        amount = player.resources[resource]
                        player.remove_resources({resource: amount})
                        current_player.add_resources({resource: amount})
                        print("Took %d from player %d" % (amount, i + 1))
        if game.get_victory_points(current_player) >= 10:
            print("Congratulations! Player %d wins!" % (current_player_num + 1))
            print("Final board:")
            print(game.board)
            sys.exit(0)

    current_player_num = (current_player_num + 1) % len(game.players)
