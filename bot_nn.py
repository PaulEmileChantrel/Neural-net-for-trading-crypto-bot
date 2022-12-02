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
    LSTM(4),
    Dense(1,activation='linear')

],name='arbs_bot_v1')

model.compile(
    loss=tf.keras.losses.MeanSquaredError(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
)

history = model.fit(
    X_train,y_train,
    epochs=2,
    validation_data=(X_test,y_test)
)

model.save('arbs_bot_v1')
plot_loss_tf(history)

def unnormalize(scaler,vec):
    vec = np.c_[vec,np.zeros(vec.shape[0])]
    vec = scaler.inverse_transform(vec)
    vec = [x[0] for x in vec]
    return vec


def check_prediction_accuracy(model,scaler,X_test,y_test,n):
    last_day = X_test[:,n-1,0]
    last_day = unnormalize(scaler,last_day)
    prediction = unnormalize(scaler,model.predict(X_test))
    y = unnormalize(scaler,y_test)
    correct_prediction = 0
    corr_variation = []
    incorrect_prediction = 0
    incorr_variation = []
    total = len(prediction)
    for i in range(total):
        pred = prediction[i]-last_day[i]
        reality = y[i] -last_day[i]
        variation = y[i]/last_day[i]-1
        if pred*reality >0:#correct direction
            correct_prediction+=1

            corr_variation.append(abs(variation))
        elif pred*reality <0:
            incorrect_prediction+=1
            
            incorr_variation.append(abs(variation))
    corr_variation_mean = np.mean(np.array([corr_variation]))*100
    incorr_variation_mean = np.mean(np.array([incorr_variation]))*100
    corr_pct = correct_prediction/total*100
    incorr_pct = incorrect_prediction/total*100

    print(f'The model was correct {corr_pct:0.2f}% of the time ({correct_prediction} out of {total}) with an average variation of {corr_variation_mean}%')
    print(f'The model was incorrect {incorr_pct:0.2f}% of the time ({incorrect_prediction} out of {total}) with an average variation of {incorr_variation_mean}%')


def trade_with_net(model,scaler,X_test,y_test,n):
    cash = 100000
    btc = 0
    stop_loss = 0.1 #10%
    fee = 0.0001    #0.01%
    last_day = X_test[:,n-1,0]
    last_day = unnormalize(scaler,last_day)
    prediction = unnormalize(scaler,model.predict(X_test))
    y_test = unnormalize(scaler,y_test)

    #trading rules
    # 1) If you own cash, all in if we predict tmr will be up
    # 2) If you own cash, do nothing if we predict tmr will be down
    # 3) If you own btc, sell all if we predict tmr is down OR if today was down and we hit the stop loss
    # 4) If you own btc, do nothing if we predict tmr will be up

    for i in range(len(prediction)):

        if cash!=0 : #We have cash
            if prediction[i]-last_day[i]>0: #We predict the stock will go up
                btc = cash/last_day[i]*(1-fee)
                bought_price = last_day[i]
                cash_stop = cash*(1-stop_loss) #If the cash amount of btc goes bellow, we will sell
                cash = 0
        else : #We own btc
            if bought_price/last_day[i]<=stop_loss: # stop loss hit during the day
                cash = cash_stop
                btc = 0
            elif prediction[i]-last_day[i]<0:#We predict it will go down
                cash = last_day[i]*btc
                btc = 0
    if cash==0:
        cash = last_day[-1]*btc
    print(cash)


check_prediction_accuracy(model,scaler,X_test,y_test,n)
trade_with_net(model,scaler,X_test,y_test,n)
plt.show()
