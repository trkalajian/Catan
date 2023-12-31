from sympy.logic.inference import valid
from pycatan.board import BuildingType
from pycatan import DevelopmentCard
from argparse import Action
import math
import random
from features import Features
import numpy as np
import numdifftools as nd
from agent_files import agent
from os import listdir, path
from os.path import isfile, join
import glob
import time
import pandas as pd


class ActorCritic(agent.HeuristicAgent):
    gamma = 0.9
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
        self.oldScore = 0
        #w is the action weights
        self.newEpisode = True
        self.previousState = None
        self.previousAction = None
        self.previousAllowedActions = []
        #thetaSize = (len(currentStateFlat) + 1)*self.numActions

        myPath = "results/theta/"
        #thetaFiles = [f for f in listdir(myPath) if isfile(join(myPath, f))]
        thetaFiles = glob.glob(str(myPath)+'*')
        latestTheta = max(thetaFiles, key=path.getctime)
        print(myPath)
        print(latestTheta)
        print("Loading: " + str(latestTheta))
        self.theta = pd.read_pickle(str(latestTheta))
        #self.theta = pickle.load(str(myPath) + str(latestTheta))
        #self.theta = np.zeros(thetaSize)
#         self.theta = [-4.05544693e-01, -4.05545378e-01, -1.35181736e-01, -1.34821400e+02,
#  -1.35181907e-01, -4.13055827e-02, -1.35181564e-01, -1.26977099e+02,
#  -2.70363128e-01, -1.50201928e-02, -1.35181736e-01, -1.60453394e+02,
#  -2.70363814e-01, -1.12651589e-02, -2.70363643e-01, -2.00365858e+02,
#  -1.35181564e-01, -3.00403524e-02,  0.00000000e+00, -1.52826870e+02,
#   0.00000000e+00,  0.00000000e+00, -1.35181736e-01,  1.18513773e+00,
#   1.77770660e+00,  5.92568865e-01,  1.75301623e+01,  5.92568865e-01,
#   4.93807388e-02,  0.00000000e+00,  2.40154993e+01,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00,  2.43117837e+01,  0.00000000e+00,
#   6.58409850e-02,  2.96284433e+00,  3.42208520e+01,  1.77770660e+00,
#   1.97522955e-01,  1.77770660e+00,  3.08958822e+01,  5.92568865e-01,
#   8.23012312e-02,  5.92568865e-01, -1.11146363e+00, -1.11146363e+00,
#  -5.55731815e-01, -2.86819365e+01, -1.66719545e+00, -9.26219692e-02,
#  -1.11146363e+00, -2.26769455e+01, -1.11146363e+00, -1.54369949e-01,
#  -5.55731815e-01, -2.28776264e+01, -5.55731815e-01, -6.17479795e-02,
#  -5.55731815e-01, -5.47087098e+01,  0.00000000e+00,  0.00000000e+00,
#  -5.55731815e-01, -4.61874886e+01, -5.55731815e-01, -7.71849743e-02,
#  -5.55731815e-01,  1.11146363e+00,  1.11146363e+00,  5.55731815e-01,
#   2.86819365e+01,  1.66719545e+00,  9.26219692e-02,  1.11146363e+00,
#   2.26769455e+01,  1.11146363e+00,  1.54369949e-01,  5.55731815e-01,
#   2.28776264e+01,  5.55731815e-01,  6.17479795e-02,  5.55731815e-01,
#   5.47087098e+01,  0.00000000e+00,  0.00000000e+00,  5.55731815e-01,
#   4.61874886e+01,  5.55731815e-01,  7.71849743e-02,  5.55731815e-01,
#   1.18513773e+00,  1.77770660e+00,  5.92568865e-01,  1.75301623e+01,
#   5.92568865e-01,  4.93807388e-02,  0.00000000e+00,  2.40154993e+01,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  2.43117837e+01,
#   0.00000000e+00,  6.58409850e-02,  2.96284433e+00,  3.42208520e+01,
#   1.77770660e+00,  1.97522955e-01,  1.77770660e+00,  3.08958822e+01,
#   5.92568865e-01,  8.23012312e-02,  5.92568865e-01,  4.05544693e-01,
#   4.05545378e-01,  1.35181736e-01,  1.34821400e+02,  1.35181907e-01,
#   4.13055827e-02,  1.35181564e-01,  1.26977099e+02,  2.70363128e-01,
#   1.50201928e-02,  1.35181736e-01,  1.60453394e+02,  2.70363814e-01,
#   1.12651589e-02,  2.70363643e-01,  2.00365858e+02,  1.35181564e-01,
#   3.00403524e-02,  0.00000000e+00,  1.52826870e+02,  0.00000000e+00,
#   0.00000000e+00,  1.35181736e-01, -2.37027546e+00, -3.55541319e+00,
#  -1.18513773e+00, -3.50603245e+01, -1.18513773e+00, -9.87614775e-02,
#   0.00000000e+00, -4.80309986e+01,  0.00000000e+00,  0.00000000e+00,
#   0.00000000e+00, -4.86235674e+01,  0.00000000e+00, -1.31681970e-01,
#  -5.92568865e+00, -6.84417039e+01, -3.55541319e+00, -3.95045910e-01,
#  -3.55541319e+00, -6.17917644e+01, -1.18513773e+00, -1.64602462e-01,
#  -1.18513773e+00,]

        return
    
    def initializeEpisode(self, game, player):
        self.player = player
        if self.firstInitialization:
            currentState = Features(game, self.player)
            #currentStateFlat = currentState.flattenFeature(game, self.player)
            self.w=np.zeros(2)
            #self.w = np.zeros(len(currentStateFlat) + 1)
        self.firstInitialization = False
        self.newEpisode = True





        return 
    
    def terminateEpisode(self, finalReward):
        delta = finalReward - self.valueFunction(self.previousState, self.w)

        diffValue = lambda w : self.valueFunction(self.previousState, w)
        valueGradient = nd.Gradient(diffValue)(self.w)
        diffTheta = lambda theta : math.log(self.actionProbabilityPolicyAC(self.previousAction, self.previousAllowedActions, self.previousState, theta))
        thetaGradient = nd.Gradient(diffTheta)(self.theta)
        self.w =  np.add(self.w, np.multiply(valueGradient, (self.alphaW*delta)))
        self.theta = np.add(self.theta, np.multiply(thetaGradient, (self.alphaTheta*self.fallOff*delta)))
        self.fallOff = self.fallOff*self.gamma
        return
    
    def extractActionWeightFromTheta(self, currentAction, theta):
        #extracts a subsection of theta that is relevant to the current action
        sliceThetaSize = len(theta)/self.numActions
        assert int(sliceThetaSize) == sliceThetaSize
        sliceThetaSize = int(sliceThetaSize)
        actionTheta = theta[currentAction*sliceThetaSize:currentAction*sliceThetaSize+sliceThetaSize]
        return actionTheta
    
    def chooseAction(self, game, allowedActions, reward):
        currentState = Features(game, self.player).flattenFeature(game, self.player)
        if self.newEpisode:
            self.fallOff = 1
            self.newEpisode = False
            action = self.actionSelectionPolicyAC(allowedActions, currentState, self.theta)
            self.previousState = currentState
            self.previousAction = action
            self.previousAllowedActions = allowedActions.copy()
            #print("new episode")
            return action
        else:
            #print("reward :" + str(reward))

            delta = reward + self.gamma* self.valueFunction(currentState, self.w) - self.valueFunction(self.previousState, self.w)
            #print("value current state " + str(self.valueFunction(currentState, self.w)))
            #print("value prev state " + str(self.valueFunction(self.previousState, self.w)))
                  
            #print("W " + str(self.w))
            #print("DELTA " + str(delta))
            diffValue = lambda w : self.valueFunction(self.previousState, w)
            valueGradient = nd.Gradient(diffValue)(self.w)
            #print("VALUE GRAD " + str(valueGradient))
            diffTheta = lambda theta : math.log(self.actionProbabilityPolicyAC(self.previousAction, self.previousAllowedActions, self.previousState, theta))
            thetaGradient = nd.Gradient(diffTheta)(self.theta)
            #print("THETA GRAD " + str(thetaGradient))
            self.w =  np.add(self.w, np.multiply(valueGradient, self.alphaW*delta))
            #print("NP MULT W " + str(np.multiply(valueGradient, self.alphaW*delta)))
            self.theta = np.add(self.theta, np.multiply(thetaGradient, self.alphaTheta*self.fallOff*delta))
            #print("NP MULT " + str(np.multiply(thetaGradient, self.alphaTheta*self.fallOff)))
            
            self.fallOff = self.fallOff*self.gamma
            self.previousState = currentState
            action = self.actionSelectionPolicyAC(allowedActions, currentState, self.theta)
            self.previousAction = action
            self.previousAllowedActions = allowedActions.copy()
            #print(len(self.theta))
            #print(self.theta)
            return action
        
        #we should never get here
        return None
    
    def valueFunction(self, state, w):
        valueSum = 0
        for i in range(len(w)):
            if i == (len(w) - 1):
                valueSum += w[i]
            else:
                valueSum += state[i]*w[i]
        #print("VALUE SUM " + str(valueSum))


        #valueSum = math.sqrt(valueSum)
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
        if weightSum > 350:
            return 350
        if weightSum < -350:
            return -350
        return weightSum
    
    def actionSelectionPolicyAC(self, allowedActions, currentState, theta):
        probabilityForAction = []
        probabilityForActionSum = []
        #find the action to take based on probability
        for i in range(len(allowedActions)):
            try:
                probabilityForAction.append(math.exp(self.h(currentState, allowedActions[i], theta)))
            except:
                print(max(currentState))
                print(sum(currentState))
                thetaArray= self.extractActionWeightFromTheta(allowedActions[i], theta)
                print("currentstate size: " + str(len(currentState)))
                print("Theta Size: " + str(len(theta)))
                print(self.h(currentState, allowedActions[i], theta))

                for j in range(len(currentState)):
                    print(str(j) + " : " + str(thetaArray[j]))
                    print(str(j) +" : " + str(currentState[j]))
                print(theta[j+1])
        

                raise Exception("Overflow")

        probSum = np.sum(probabilityForAction)
        for prob in probabilityForAction:
            probabilityForActionSum.append(prob/probSum)
        selection = random.random()

        
        for i in range(len(probabilityForActionSum) - 1):
            if selection <= probabilityForActionSum[i]:
                #print("allowed action from policy: " + str(allowedActions[i]))

                return allowedActions[i]
            else:
                selection -= probabilityForActionSum[i]
        #print("allowed action from policy: " + str(allowedActions[len(probabilityForActionSum)-1]))
        return allowedActions[len(probabilityForActionSum)-1]
    

    def actionProbabilityPolicyAC(self, actionForProb, allowedActions, currentState, theta):
        #returns the probability for an action to be taken
        probabilityForAction = []
        probabilityForActionSum = []
        actionIndex = None
        #print(theta)

        for i in range(len(allowedActions)):
            if allowedActions[i] == actionForProb:
                actionIndex = i
            try:
                probabilityForAction.append(math.exp(self.h(currentState, allowedActions[i], theta)))
            except:
                print(theta)
                raise Exception("Overflow")
        probSum = np.sum(probabilityForAction)
        result = probabilityForAction[actionIndex]/probSum
        if result <= 0 or math.isnan(result):
            print("allowed actions: " + str(allowedActions))
            print("action index prob: " + str(probabilityForAction[actionIndex]))
            print("Prob sum:" + str(probSum))
            print("prob result: " + str(result))
            raise Exception("Invalid Prob")
        return result
    
    def policy(self, game, current_player_num=None):
        #newScore = (game.get_victory_points(self.player) - 2) *0.1
        #reward = newScore - self.oldScore
        # if self.previousAction != 6:
        #      reward += 2
        # if self.previousAction == 6  and len(self.previousAllowedActions) > 1:
        #      reward -= 10
        reward = 0
        #self.oldScore = newScore
        validActions = []
        if self.player.has_resources(BuildingType.SETTLEMENT.get_required_resources()) and game.board.get_valid_settlement_coords(self.player):
            validActions.append(self.BUILD_SETTLEMENT)
        if self.player.has_resources(BuildingType.CITY.get_required_resources()) and game.board.get_valid_city_coords(self.player):
            validActions.append(self.BUILD_CITY)
        if self.player.has_resources(BuildingType.ROAD.get_required_resources()) and game.board.get_valid_road_coords(self.player) and self.player.num_roads > 0:
            validActions.append(self.BUILD_ROAD)
        if self.player.has_resources(DevelopmentCard.get_required_resources()) and game.development_card_deck:
            validActions.append(self.BUILD_CARD)
        if self.player.get_possible_trades():
            validActions.append(self.MAKE_TRADE)
        if DevelopmentCard.KNIGHT in [card for card, amount in self.player.development_cards.items() if
                             amount > 0]:
            validActions.append(self.PLAY_KNIGHT)
        validActions.append(self.PASS_TURN)
        #print("these are valid actions: " + str(validActions))
        choosenAction = self.chooseAction(game, validActions, reward)
     
        #print("AC chooses: " + str(choosenAction))
        if choosenAction == self.BUILD_SETTLEMENT:
            return [1, 1]
        if choosenAction == self.BUILD_CITY:
            return [1, 2]
        if choosenAction == self.BUILD_ROAD:
            return [1, 3]
        if choosenAction == self.BUILD_CARD:
            return [1 ,4]
        if choosenAction == self.MAKE_TRADE:
            return [2, None]
        if choosenAction == self.PLAY_KNIGHT:
            return [3, None]
        if choosenAction == self.PASS_TURN:
            return [4, None]
            
        
        #build settlement: [1,1]
#build city: [1, 2]
#build road: [1, 3]
#build dev_card: [1,4]
#play knight: [3, None]
#trade resource: [2, None]
#pass: [4, None]

    #         BUILD_SETTLEMENT = 0
    # BUILD_CITY = 1
    # BUILD_ROAD = 2
    # BUILD_CARD = 3
    # MAKE_TRADE = 4
    # PLAY_KNIGHT = 5
    # PASS_TURN = 6

