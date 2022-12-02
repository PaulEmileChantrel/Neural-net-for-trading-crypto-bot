import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,GRU
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

#Data preprocessing
df = pd.read_csv('btc_preprocessed1.csv')
data = df[['binance_btc_closing_price','coinbase_btc_closing_price','binance_log_returns','coinbase_log_returns']].to_numpy()
#
scaler = MinMaxScaler(feature_range=(0,1)).fit(data)
normalized_data = scaler.transform(data)

data_len = normalized_data.shape[0]
new_data_len = data_len - data_len%1440


normalized_data = normalized_data[:new_data_len,:] #We take of the extra data, with no full day at the end
normalized_data = normalized_data.reshape(-1,1440,4)
#print(data)
print(normalized_data.shape)

## Action Space :
# 6 actions
# Buy y1 BTC on Binance -> 0
# Sell y2 BTC on Binance -> 1
# Do nothing on Binance -> 2
# Buy y3 BTC on Coinbase -> 3
# Sell y4 BTC on Coinbase -> 4
# Do nothing on Coinbase -> 5
# Actions on 1 exchange are discreate (you can't buy and sell at the same time on the same exchange)
# y1 and y3 varies between 0 and the max num of possible btc buy with the current amount max_num = cash/btc_price
# y2 and y4 varies between 0 and the number of btc

## Observations space :
# Number of $ (100,000 at t=0)
# Number of BTC (0 at t=0)
# Binance current price
# Coinbase currrent price
# (We considere that $ and BTC can move freely and instantly between Binance and Coinbase for now)


## Rewards :
# If you earn $x after at the end of a min 1*x
# If you loose $x after at the end of a min -1.5*x
# If you make a trade of size y -0.001*y


## Episode Termination
# Portfolio goes to 0
# End of day

X = np.load('btc.npy')
y = np.zeros(shape = X.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15, random_state=1)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


model = Sequential([
    tf.keras.Input(shape=(2,1440,X.shape[0])),
    GRU(128,return_sequences=True),
    GRU(128),
    Dense(6,activation='linear')

],name='arbs_bot')

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
)
