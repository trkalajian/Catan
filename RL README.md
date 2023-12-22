Authors: Gyde Lund and Taylor Kalajian
Class: CS138 - RL Tufts Univerisity
Catan Final Project

SRC Directory contains the following:

Packages Required to be up to day as of DEC 2023: pickle, PyTorch, Numpy, Colored (for board print outs), numdifftools

Main Training Loop: train.py
Use train.py to run a training session. From the file select what agent types you are running, how many players, and other game settings. Will save progress to a folder called results

agent_files directory contains files relevant to the DQN and heuristic Agents (actor critic had its own structure)

features contains functions to extract and normalize features for calculating the state-action space

pycatan directory contains the game files

methods.py includes various game methods removed from train.py to make room for the training code. 

Play.py will run a game with human inputs

