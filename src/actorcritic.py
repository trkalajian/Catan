from pycatan.board import BuildingType
from pycatan import DevelopmentCard
from argparse import Action
import math
import random
from features import Features
import numpy as np
import numdifftools as nd
from agent_files import heuristics
from agent_files import agent

class ActorCritic(agent.HeuristicAgent):
    gamma = 0.7
    #actions
    #build settlement
    #build road
    #build city
    #build card
    #make trade
    #play knight
    #pass turn

    BUILD_SETTLEMENT = 0
    BUILD_CITY = 1
    BUILD_ROAD = 2
    BUILD_CARD = 3
    MAKE_TRADE = 4
    PLAY_KNIGHT = 5
    PASS_TURN = 6
    
    #number of actions for diving theta into subsets
    numActions = 7
    
    def __init__(self, place_settlement_func, place_road_func, place_robber_func, choose_best_trade, place_city_func):
        super().__init__(self.policy, place_settlement_func, place_road_func, place_robber_func, choose_best_trade, place_city_func)
        self.fallOff = None
        #theta are the policyweights
        self.firstInitialization = True
        self.alphaTheta = .1
        self.alphaW = .1
        
        #w is the action weights
        self.newEpisode = True
        self.previousState = None
        self.previousAction = None
        self.previousAllowedActions = None

        return
    
    def initializeEpisode(self, game, player):
        self.player = player
        if self.firstInitialization:
            currentState = Features(game, self.player)
            currentStateFlat = currentState.flattenFeature(game, self.player)

            thetaSize = (len(currentStateFlat) + 1)*self.numActions

            self.theta = np.zeros(thetaSize)
            self.w = np.zeros(len(currentStateFlat) + 1)
        self.firstInitialization = False
        self.newEpisode = True





        return 
    
    def terminateEpisode(self, finalReward):
        delta = finalReward - self.valueFunction(self.previousState, self.w)

        diffValue = lambda w : self.valueFunction(self.previousState, w)
        valueGradient = nd.Gradient(diffValue)(self.w)
        diffTheta = lambda theta : math.log(self.actionProbabilityPolicyAC(self.previousAction, self.previousAllowedActions, self.previousState, theta))
        thetaGradient = nd.Gradient(diffTheta)(self.theta)
        self.w =  np.add(self.w, np.multiply(valueGradient, self.alphaW*delta))
        self.theta = np.add(self.theta, np.multiply(thetaGradient, self.alphaTheta*self.fallOff))
        self.fallOff = self.fallOff*self.gamma
        return
    
    def extractActionWeightFromTheta(self, currentAction, theta):
        #extracts a subsection of theta that is relevant to the current action
        sliceThetaSize = len(theta)/self.numActions
        assert int(sliceThetaSize) == sliceThetaSize
        actionTheta = theta[currentAction*sliceThetaSize:currentAction*sliceThetaSize+sliceThetaSize]
        return actionTheta
    
    def chooseAction(self, game, allowedActions, reward=None):
        currentState = Features(game, self.player).flattenFeature(game, self.player)
        if self.newEpisode:
            self.fallOff = 1
            newEpisode = False
            action = self.actionSelectionPolicyAC(allowedActions, currentState, self.theta)
            self.previousState = currentState
            self.previousAction = action
            self.previousAllowedActions = allowedActions.copy()
            return action
        else:
            delta = reward + self.gamma* self.valueFunction(currentState, self.w) - self.valueFunction(self.previousState, self.w)
            diffValue = lambda w : self.valueFunction(self.previousState, w)
            valueGradient = nd.Gradient(diffValue)(self.w)
            diffTheta = lambda theta : math.log(self.actionProbabilityPolicyAC(self.previousAction, self.previousAllowedActions, self.previousState, theta))
            thetaGradient = nd.Gradient(diffTheta)(self.theta)
            self.w =  np.add(self.w, np.multiply(valueGradient, self.alphaW*delta))
            self.theta = np.add(self.theta, np.multiply(thetaGradient, self.alphaTheta*self.fallOff))
            
            self.fallOff = self.fallOff*self.gamma
            self.previousState = currentState
            action = self.actionSelectionPolicyAC(allowedActions, currentState, self.theta)
            self.previousAction = action
            self.previousAllowedActions = allowedActions.copy()
            return action
        
        #we should never get here
        return None
    
    def valueFunction(self, state, w):
        valueSum = 0
        for i in range(len(w)):
            if i == len(w) - 1:
                valueSum += w[i]
            else:
                valueSum += state[i]*w[i]
        return valueSum
    
    def h(self, state, action, theta):
        #h function produces a proportional weight that guarantees everything to be greater than 0 percent and less than 100 percent.  Guarantees exploration
        weightSum = 0
        extractedTheta = self.extractActionWeightFromTheta(action, theta)
        for i in range(len(extractedTheta)):
            if i == len(extractedTheta)  - 1:
                weightSum += extractedTheta[i]
            else:
                weightSum += extractedTheta[i]*state[i]
        return weightSum
    
    def actionSelectionPolicyAC(self, allowedActions, currentState, theta):
        probabilityForAction = []
        probabilityForActionSum = []
        #find the action to take based on probability
        for i in range(len(allowedActions)):
            probabilityForAction.append(math.e**self.h(currentState, allowedActions[i], theta))
        probSum = np.sum(probabilityForAction)
        for prob in probabilityForAction:
            probabilityForActionSum.append(probabilityForAction[prob]/probSum)
        selection = math.random()
        for i in probabilityForActionSum[i]:
            if selection <= probabilityForActionSum[i]:
                return allowedActions[i]
            else:
                selection -= probabilityForActionSum[i]
        #we should not get here
        assert False
        return None
    

    def actionProbabilityPolicyAC(self, actionForProb, allowedActions, currentState, theta):
        #returns the probability for an action to be taken
        probabilityForAction = []
        probabilityForActionSum = []
        actionIndex = None
        for i in range(len(allowedActions)):
            if allowedActions[i] == actionForProb:
                actionIndex = i
            probabilityForAction.append(math.e**self.h(currentState, allowedActions[i], theta))
        probSum = np.sum(probabilityForAction)
        return probabilityForAction[actionIndex]/probSum
    
    def policy(self, game):
        validActions = []
        if self.player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) and game.board.get_valid_settlement_coords(self.player):
            validActions.append(self.BUILD_SETTLEMENT)
        if self.player.has_resources(BuildingType.CITY.get_required_resources()) and game.board.get_valid_city_coords(self.player):
            validActions.append(self.BUILD_CITY)
        if self.player.has_resources(BuildingType.ROAD.get_required_resources()) and game.board.get_valid_road_coords(self.player):
            validActions.append(self.BUILD_ROAD)
        if self.player.has_resources(DevelopmentCard.get_required_resources()):
            validActions.append(self.BUILD_CARD)
        if self.player.get_possible_trades():
            validActions.append(self.MAKE_TRADE)
        if DevelopmentCard.KNIGHT in [card for card, amount in self.player.development_cards.items() if
                             amount > 0]:
            validActions.append(self.PLAY_KNIGHT)
        validActions.append(self.PASS_TURN)

        self.chooseAction(game, validActions)
        


    #         BUILD_SETTLEMENT = 0
    # BUILD_CITY = 1
    # BUILD_ROAD = 2
    # BUILD_CARD = 3
    # MAKE_TRADE = 4
    # PLAY_KNIGHT = 5
    # PASS_TURN = 6

