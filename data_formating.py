import pandas as pd
import numpy as np
from tempfile import TemporaryFile

#1) We want to erase the difference between the two exchanges price
# Since Binance give the BTC/USDT and Coinbase give the BTC/USD, we want to make sure that we dont have a small shift in all our data

df = pd.read_csv('btc_data.csv')
df.fillna('',inplace=True)
df = df[df['coinbase_btc_closing_price']!='']
df['diff'] = df['binance_btc_closing_price'].astype('float32')-df['coinbase_btc_closing_price'].astype('float32')
m = df['diff'].mean()#In average, the difference should be $0
print(m)

#We have m = -1.669 $ so we subtract it to coinbase price
df['coinbase_btc_closing_price'] = df['coinbase_btc_closing_price'] + m



#2) We want to break the data into 1440 subset

data = df[['binance_btc_closing_price','coinbase_btc_closing_price']].to_numpy()
#
data_len = data.shape[0]
new_data_len = data_len - data_len%1440


data = data[:new_data_len,:] #We take of the extra data, with no full day at the end
data = data.reshape(-1,1440,2)
#print(data)
print(data.shape)


with open('btc.npy', 'wb') as f:
    np.save(f,data)
