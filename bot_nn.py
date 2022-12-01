import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.model_selection import train_test_split

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *



MEMORY_SIZE = 100_000 # size of memory buffer.
GAMMA = 0.995 # discount factor.
ALPHA = 1e-3 # learning rate.
NUM_STEPS_FOR_UPDATE = 4 # perform a learning update every C time steps.



## Action Space :
# 4 actions
# Buy y1 BTC on Binance -> 0
# Sell y2 BTC on Binance -> 1
# Buy y3 BTC on Coinbase -> 2
# Sell y4 BTC on Coinbase -> 3
# Actions on 1 exchange are discreate (you can't buy and sell at the same time on the same exchange)
# y1 and y3 varies between 0 and the max num of possible btc buy with the current amount max_num = cash/btc_price
# y2 and y4 varies between 0 and the number of btc

## Observations space :
# Number of $ (100,000 at t=0)
# Number of BTC (0 at t=0)
# (We considere that $ and BTC can move freely and instantly between Binance and Coinbase for now)


## Rewards :
# If you earn $x after at the end of a min 1*x
# If you loose $x after at the end of a min -1.5*x
# If you make a trade of size y -0.001*y


## Episode Termination
# Portfolio goes to 0
# End of day
