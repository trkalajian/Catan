import numpy as np
from pycatan import Game, Player, Resource
from pycatan.board import BuildingType, Board

# class helps to construct features for training
# the feature array contains all of the data in a flattened and consistent form if needed

class Features():
    # array of feature information, use this for training
    featuresArray = []
    RESOURCE_ORDER = [Resource.WOOL, Resource.GRAIN, Resource.LUMBER, Resource.BRICK, Resource.ORE]

    # - resources of self and other players
    currentPlayerResources = None
    # opponent resources are a dict: players|resources
    opponentResources = {}

    # - number of available settlement spots for each player

    currentPlayerSettlementSpots = None
    # opponent settlements are a dict: players|coordinates
    opponentPlayerSettlementSpots = {}

    # - average yield for a roll for each player
    # current player average yield is a dict: resources|yield
    averageYieldCurrentPlayer = {}

    # opposing player avg yield is a dict:players and resources | yield
    averageYieldOpposingPlayers = {}

    # Victory points for each player
    # current player VP
    currentPlayerVictoryPoints = None

    # dict of op|VP
    opponentVictoryPoints = {}

    # dict of coordinates|resources
    resourceLocations = {}

    intersection_states = []

    def __init__(self, game: Game, player: Player):
        for i in range(len(game.players)):      # establishing vps and resources

            if game.players[i] == player:
                self.currentPlayerResources = player.resources
                self.currentPlayerSettlementSpots = game.board.get_valid_settlement_coords(player)
                self.currentPlayerVictoryPoints = game.get_victory_points(player)
            else:
                self.averageYieldOpposingPlayers[game.players[i]] = {}
                self.opponentVictoryPoints[game.players[i]] = game.get_victory_points(game.players[i])
                self.opponentResources[game.players[i]] = (game.players[i].resources)
                self.opponentPlayerSettlementSpots[game.players[i]] = (
                    game.board.get_valid_settlement_coords(game.players[i]))
                # Initialize intersection_states
                self.intersection_states = np.zeros((54))  # 54 intersections, 8 features each
                
        # UNCOMMENT FOR INTERSECTION FEATURES. CHANGES FLATTENED FEATURTED FROM 44 TO 98
        # for i, coord in enumerate(game.board.intersections.keys()):
        #     intersection = game.board.intersections[coord]
        #     connected_hexes = game.board.get_hexes_connected_to_intersection(coord)
        #     hex_resources = game.board.get_hex_resources_for_intersection(coord)
        #     yield_sum = 0
        #
        #     # Encode the building information
        #     if intersection.building is not None:
        #         if intersection.building.owner == player:
        #             # Player-owned settlement or city
        #             self.intersection_states[i] = 1
        #             # self.intersection_states[i][1] = 1 if intersection.building.building_type == BuildingType.SETTLEMENT else 2
        #         else:
        #             # Opponent-owned settlement or city
        #             self.intersection_states[i] = -1
        #             # self.intersection_states[i][1] = -1 if intersection.building.building_type == BuildingType.SETTLEMENT else -2
        #     else:
        #         # Empty intersections
        #         self.intersection_states[i] = 0
        #         # self.intersection_states[i][1] = 0


        # Establishing yields
        for roll in range(2, 13):
            if roll < 8:
                rollProbability = (roll - 1) / 36
            else:
                rollProbability = (13 - roll) / 36

            currentYields = game.board.get_yield_for_roll(roll)
            for playerYield in currentYields:
                if playerYield == player:
                    for resource in currentYields[player].total_yield:

                        if resource in self.averageYieldCurrentPlayer:
                            self.averageYieldCurrentPlayer[resource] += currentYields[player].total_yield[
                                                                            resource] * rollProbability
                        else:
                            self.averageYieldCurrentPlayer[resource] = currentYields[player].total_yield[
                                                                           resource] * rollProbability
                else:
                    for resource in currentYields[playerYield].total_yield:
                        if resource in self.averageYieldOpposingPlayers[playerYield]:
                            self.averageYieldOpposingPlayers[playerYield][resource] += \
                            currentYields[playerYield].total_yield[resource] * rollProbability
                        else:
                            self.averageYieldOpposingPlayers[playerYield][resource] = \
                            currentYields[playerYield].total_yield[resource] * rollProbability

                            # for coord in game.board.intersections:
        #     coordResources = game.board.get_hex_resources_for_intersection(coord)

        # hexTokens = game.board.get_hexes_connected_to_intersection(coord)
        # self.resourceLocations[coord] = (coordResources)

    def flattenFeature(self, game: Game, player: Player):
        self.featuresArray = []
        # function to flatten features so that they are in a consistent order and consumable by an agent
        # feature order player VP, op VP, player resources, player yield, op resources, op yield,
        self.featuresArray.append(self.currentPlayerVictoryPoints)
        for plr in game.players:
            if plr != player:
                self.featuresArray.append(self.opponentVictoryPoints[plr])

        for resource in Resource:
            self.featuresArray.append(self.currentPlayerResources[resource])
        for resource in Resource:
            if resource in self.averageYieldCurrentPlayer:
                self.featuresArray.append(self.averageYieldCurrentPlayer[resource])
            else:
                self.featuresArray.append(0)

        for plr in game.players:
            if plr != player:
                for resource in Resource:
                    self.featuresArray.append(self.opponentResources[plr][resource])
                    if resource in self.averageYieldOpposingPlayers[plr]:
                        self.featuresArray.append(self.averageYieldOpposingPlayers[plr][resource])
                    else:
                        self.featuresArray.append(0)
        
        # UNCOMMENT FOR INTERSECTION FEATURES. CHANGES FLATTENED FEATURTED FROM 44 TO 98
        # flattened_intersections = self.intersection_states.flatten() 
        # self.featuresArray.extend(flattened_intersections)

        return self.featuresArray

    def _calculate_yield_probability(self, hex_token):
        # Calculate yield probability based on token number
        if hex_token in [2, 12]:
            return 1 / 36
        elif hex_token in [3, 11]:
            return 2 / 36
        elif hex_token in [4, 10]:
            return 3 / 36
        elif hex_token in [5, 9]:
            return 4 / 36
        elif hex_token in [6, 8]:
            return 5 / 36
        else:
            return 0
