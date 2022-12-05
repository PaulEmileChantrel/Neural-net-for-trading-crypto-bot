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
#df = pd.read_csv('btc_preprocessed1.csv')
df = pd.read_csv('btc_prossessed_V1p5.csv')
data = df[['b_close','b_log_ret','b_macd','b_rsi_7','b_rsi_21','b_sma_7','b_sma_21','b_wr_7','b_wr_21','b_ups_7_c','b_ups_14_c','b_downs_7_c','b_downs_14_c']].to_numpy()
#
#data = df[['binance_btc_closing_price','binance_log_returns']].to_numpy()


number_of_points = 200000 #we reduce the dataset size
data = data[data.shape[0]-number_of_points:,:]
scaler = MinMaxScaler(feature_range=(0,1)).fit(data)
normalized_data = scaler.transform(data)
print(normalized_data.shape)
n = 7

#We don't want common points between the training and testing dataset
train = normalized_data[:int(normalized_data.shape[0]*0.8),:]
test = normalized_data[int(normalized_data.shape[0]*0.8):,:]
# plt.plot(train['binance_btc_closing_price'])
# plt.plot(test['binance_btc_closing_price'])
# plt.legend(['training','testing'])
# plt.xlabel('Time (s)')
# plt.ylabel('Bitcoin price ($)')
# plt.title('Bitcoin price as a function of time')
# plt.show()
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
    tf.keras.Input(shape=(n,X_train.shape[2])),
    GRU(128,activation='relu',return_sequences=True),
    #LSTM(32,return_sequences=True),
    GRU(128,activation='relu'),
    Dense(32,activation='relu'),
    Dense(16,activation='relu'),
    Dense(1,activation='linear')

],name='arbs_bot_v1')

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    metrics=['mean_absolute_error']
)
#We check the results without training the model
#recursive_prediction(model,scaler,X_test,y_test)
# check_prediction_accuracy(model,scaler,X_test,y_test,n)
# trade_with_net(model,scaler,X_test,y_test,n)

history = model.fit(
    X_train,y_train,
    epochs=20,
    validation_data=(X_test,y_test)
)

model.save('arbs_bot_v1')
plot_loss_tf(history)



# recursive_prediction(model,scaler,X_test,y_test)
check_prediction_accuracy(model,scaler,X_train,y_train,n)
trade_with_net(model,scaler,X_train,y_train,n,False)
check_prediction_accuracy(model,scaler,X_test,y_test,n)
trade_with_net(model,scaler,X_test,y_test,n)
plt.show()
