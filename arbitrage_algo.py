#Without NN -> algorythm to do arbitrage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from autils import *
from nn_utils_function import *



df = pd.read_csv('btc_preprocessed1.csv')
data = df[['binance_btc_closing_price','coinbase_btc_closing_price']].to_numpy()
diff = np.mean(data[:,0]-data[:,1])
data[:,1] += diff
print(np.mean(data[:,0]-data[:,1]))
cash = 100000
btc = 0
fee = 0 # We assume the same fee on both exchanges

def calcul_gain(cash,fee):
    variation = []
    for i in range(data.shape[0]):
        bin_price = data[i,0]
        coin_price = data[i,1]

        #print(bin_price,coin_price,cash)
        if bin_price*((1-fee)**2)>coin_price and coin_price!=0:
            print('b:',i,bin_price,bin_price*((1-fee)**2),coin_price,cash,bin_price*((1-fee)**2)/coin_price)
            #buy on coinbase
            btc = cash/coin_price*(1-fee)
            #sell on coinbase
            cash = btc*bin_price*(1-fee)
            variation.append(bin_price/coin_price*((1-fee)**2))

            # or cash*= bin_price/coin_price
        elif coin_price*((1-fee)**2)>bin_price and bin_price!=0:
            print('c:',i,coin_price,coin_price*((1-fee)**2),bin_price,cash,coin_price*((1-fee)**2)/bin_price)

            cash *= coin_price/bin_price*((1-fee)**2)
            variation.append(coin_price/bin_price)
    print(np.mean(np.array(variation)))
    print(cash)

#calcul_gain(cash,0)
calcul_gain(cash,0.04) #0.1%
print(1.00003**100000)


### Results :
# Average variation 1.0024 (0.24%)
# If we were able to capture them all, every min, the cash goes to inf

# In reality, there is a spread between the buy and sell side and we have to account for transfer time between the two exchange
