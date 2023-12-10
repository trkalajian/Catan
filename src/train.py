from pycatan import Game
from pycatan.board import RandomBoard
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random

game = Game(RandomBoard(), 2)

pOne = game.players[0]
pTwo = game.players[1]
settlement_coords = game.board.get_valid_settlement_coords(player = pOne, ensure_connected = False)
game.build_settlement(player = pOne, coords = random.choice(list(settlement_coords)), cost_resources = False, ensure_connected = False)
game.build_settlement(player = pTwo, coords = random.choice(list(settlement_coords)), cost_resources = False, ensure_connected = False)

print(game.board)