
import numpy
from pycatan import Game, Player, Resource
from pycatan.board import Board

#class helps to construct features for training
#the feature array contains all of the data in a flattened and consistent form if needed

class Features():
    #array of feature information, use this for training
    featuresArray = []
    

    # - resources of self and other players
    currentPlayerResources = None
    #opponent resources are a dict: players|resources
    opponentResources ={}
    
    # - number of available settlement spots for each player

    currentPlayerSettlementSpots = None
    #opponent settlements are a dict: players|coordinates
    opponentPlayerSettlementSpots = {}
    
    # - average yield for a roll for each player
    #current player average yield is a dict: resources|yield
    averageYieldCurrentPlayer = {}
    
    #opposing player avg yield is a dict:players and resources | yield
    averageYieldOpposingPlayers = {}
    
    #Victory points for each player
    #current player VP
    currentPlayerVictoryPoints = None
    
    #dict of op|VP
    opponentVictoryPoints = {}
    
    #dict of coordinates|resources
    resourceLocations = {}
    

    def __init__(self, game:Game, player:Player):
        for i in range(len(game.players)):

            if game.players[i] == player:
                self.currentPlayerResources = player.resources
                self.currentPlayerSettlementSpots = game.board.get_valid_settlement_coords(player)
                self.currentPlayerVictoryPoints = game.get_victory_points(player)
            else:
                self.averageYieldOpposingPlayers[game.players[i]] = {}
                self.opponentVictoryPoints[game.players[i]] = game.get_victory_points(game.players[i])
                self.opponentResources[game.players[i]] = (game.players[i].resources)
                self.opponentPlayerSettlementSpots[game.players[i]] = (game.board.get_valid_settlement_coords(game.players[i]))
                

        for roll in range(2, 13):
            if roll < 8:
                rollProbability = (roll-1)/36
            else:
                rollProbability = (13-roll)/36
                
            currentYields = game.board.get_yield_for_roll(roll)
            for playerYield in currentYields:
                if playerYield == player:
                    for resource in currentYields[player].total_yield:
 
                        if resource in self.averageYieldCurrentPlayer:
                            self.averageYieldCurrentPlayer[resource] += currentYields[player].total_yield[resource]*rollProbability
                        else:
                            self.averageYieldCurrentPlayer[resource] = currentYields[player].total_yield[resource]*rollProbability                        
                else:
                    for resource in currentYields[playerYield].total_yield:
                        if resource in self.averageYieldOpposingPlayers[playerYield]:
                            self.averageYieldOpposingPlayers[playerYield][resource] += currentYields[playerYield].total_yield[resource]*rollProbability
                        else:
                            self.averageYieldOpposingPlayers[playerYield][resource] = currentYields[playerYield].total_yield[resource]*rollProbability    
            
            
        # for coord in game.board.intersections:
        #     coordResources = game.board.get_hex_resources_for_intersection(coord)

            #hexTokens = game.board.get_hexes_connected_to_intersection(coord)
            #self.resourceLocations[coord] = (coordResources)
    

    def flattenFeature(self, game:Game, player:Player):
        self.featuresArray = []
        #function to flatten features so that they are in a consistent order and consumable by an agent
        #feature order player VP, op VP, player resources, player yield, op resources, op yield, player settlementspots, op settlement spots, resource locations
        self.featuresArray.append(self.currentPlayerVictoryPoints)
        for plr in game.players:
                if plr != player:
                    self.featuresArray.append(self.opponentVictoryPoints[plr])

        for resource in Resource:
            self.featuresArray.append(self.currentPlayerResources[resource])
            if resource in self.averageYieldCurrentPlayer:
                self.featuresArray.append(self.averageYieldCurrentPlayer[resource])
            else:
                self.featuresArray.append(0)
            for plr in game.players:
                if plr != player:
                    self.featuresArray.append(self.opponentResources[plr][resource])
                    if resource in self.averageYieldOpposingPlayers[plr]:
                        self.featuresArray.append(self.averageYieldOpposingPlayers[plr][resource])
                    else:
                        self.featuresArray.append(0)

            
        # for coordinate in game.board.intersections:
        #     if coordinate in self.currentPlayerSettlementSpots:
        #         self.featuresArray.append(1)
        #     else:
        #         self.featuresArray.append(0)
        #     for plr in game.players:
        #         if plr != player:
        #             if coordinate in self.opponentPlayerSettlementSpots[plr]:
        #                 self.featuresArray.append(1)
        #             else:
        #                 self.featuresArray.append(0)
        
        # for coordinate in game.board.intersections:
        #     for resource in Resource:
        #         if resource in self.resourceLocations[coordinate]:
        #             self.featuresArray.append(1)
        #         else:
        #             self.featuresArray.append(0)
        return self.featuresArray
