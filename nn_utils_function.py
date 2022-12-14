
import matplotlib.pyplot as plt
import numpy as np

import copy,math

def plot_loss_tf(history):
    losses = history.history['loss']
    plt.plot(losses)
    try:
        val_loss = history.history['val_loss']
        plt.plot(val_loss)
        plt.legend(['loss','test_loss'])
    except:
        pass

    plt.xlabel('epoch')
    plt.ylabel('losses')

def unnormalize(scaler,vec,scaler_size=1):

    vec = np.c_[vec,np.zeros((vec.shape[0],scaler_size))]
    vec = scaler.inverse_transform(vec)
    vec = [x[0] for x in vec]
    return vec


def check_prediction_accuracy(model,scaler,X_test,y_test,n):
    last_day = X_test[:,n-1,0]

    last_day = unnormalize(scaler,last_day,scaler_size=X_test.shape[2]-1)
    prediction = unnormalize(scaler,model.predict(X_test),scaler_size=X_test.shape[2]-1)
    y = unnormalize(scaler,y_test,scaler_size=X_test.shape[2]-1)
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

    print(f'The model was correct {corr_pct:0.2f}% of the time ({correct_prediction} out of {total}) with an average variation of {corr_variation_mean:0.5f}%')
    print(f'The model was incorrect {incorr_pct:0.2f}% of the time ({incorrect_prediction} out of {total}) with an average variation of {incorr_variation_mean:0.5f}%')


def trade_with_net(model,scaler,X_test,y_test,n,plot=True):
    cash = 100000
    btc = 0
    stop_loss = 0.1 #10%
    fee = 0.0001    #0.01%
    last_day = X_test[:,n-1,0]
    last_day = unnormalize(scaler,last_day,scaler_size=X_test.shape[2]-1)
    prediction = unnormalize(scaler,model.predict(X_test),scaler_size=X_test.shape[2]-1)
    y_test = unnormalize(scaler,y_test,scaler_size=X_test.shape[2]-1)

    if plot:
        plt.plot(prediction)
        plt.plot(y_test)
        plt.legend(['prediction','test data'])
        plt.xlabel('Time (min)')
        plt.ylabel('Bitcoin price ($)')
        plt.title('Bitcoin price as a function of time')

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
    print(f'We have ${cash} at the end')

def recursive_prediction(model,scaler,X_test,y_test):

    prediction = []

    x = X_test[0,:,:].reshape(-1,X_test.shape[1],X_test.shape[2])

    for i in range(X_test.shape[0]):
        print(x)
        y_predict = model.predict(x,verbose=0)
        un_y = unnormalize(scaler,y_predict)
        prediction.append(un_y[0])
        x = x[:,1:,:]

        last_price = unnormalize(scaler,x[:,X_test.shape[1]-2,0])
        log_return = np.log(1+un_y[0]/last_price[0])
        log_return = np.array([[0,log_return]])
        log_return = scaler.transform(log_return)

        x = np.append(x,np.array([[[y_predict[0][0],log_return[0,1]]]]),axis=1) #If we have more parameter than the price, this won't work
        if i%(X_test.shape[0]//100)==0 and i!=0:
            print(str(round(i/X_test.shape[0]*100))+' %')
    y = unnormalize(scaler,y_test)

    plt.plot(y)
    plt.plot(prediction)
    plt.legend(['Price','Recursive prediction'])
    plt.xlabel('Time (min)')
    plt.ylabel('Bitcoin price ($)')
    plt.title('Bitcoin price Prediction')
    plt.show()
