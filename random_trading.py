import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding,GRU
from tensorflow.keras.activations import linear, relu, sigmoid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

from autils import *
from nn_utils_function import *
"""
The bot V1 will use a simple LSTM net and try to predict the next day data using the n previous day values.
We will just use Binance price for this exemple
"""

#Data preprocessing
df = pd.read_csv('btc_preprocessed1.csv')
data = df[['binance_btc_closing_price','binance_log_returns']].to_numpy()
#

number_of_points = 100000#we reduce the dataset size
data = data[data.shape[0]-number_of_points:,:]
scaler = MinMaxScaler(feature_range=(0,1)).fit(data)
normalized_data = scaler.transform(data)
print(normalized_data.shape)
n = 5

#We don't want common points between the training and testing dataset
train = normalized_data[:int(normalized_data.shape[0]*0.8),:]
test = normalized_data[int(normalized_data.shape[0]*0.8):,:]
X_train = []
X_test =[]
y_train = []
y_test = []
for i in range(n,int(normalized_data.shape[0]*0.8)):
    X_train.append(train[i-n:i,:])
    y_train.append(train[i,0])

for i in range(n,len(normalized_data)-int(normalized_data.shape[0]*0.8)):
    X_test.append(test[i-n:i,:])
    y_test.append(test[i,0])
#print(data)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array([y_train]).T
y_test = np.array([y_test]).T

# print(X_train[:5,:],y_train[:5])
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

## Neural NET
model = Sequential([
    tf.keras.Input(shape=(n,2)),
    #GRU(128,return_sequences=True),
    LSTM(8),
    Dense(1,activation='linear')

],name='arbs_bot_v1')

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
)

#We check the results without training the model
check_prediction_accuracy(model,scaler,X_test,y_test,n)
trade_with_net(model,scaler,X_test,y_test,n)
