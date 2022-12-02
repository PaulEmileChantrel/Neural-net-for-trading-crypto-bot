import pandas as pd
import numpy as np
from tempfile import TemporaryFile
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
#1) We want to erase the difference between the two exchanges price
# Since Binance give the BTC/USDT and Coinbase give the BTC/USD, we want to make sure that we dont have a small shift in all our data

df = pd.read_csv('btc_data.csv',index_col=[0])
#print(df)

df['diff'] = df['binance_btc_closing_price'].astype('float32')-df['coinbase_btc_closing_price'].astype('float32')
m = df['diff'].mean()#In average, the difference should be $0
print(m)

#We have m = -1.669 $ so we subtract it to coinbase price
df['coinbase_btc_closing_price'] = df['coinbase_btc_closing_price'] + m
#df = df.sort_index(ascending=True)

df['binance_returns'] = df['binance_btc_closing_price'].pct_change()
df['coinbase_returns'] = df['coinbase_btc_closing_price'].pct_change()


df['binance_log_returns'] = np.log(1 + df['binance_returns'])
df['coinbase_log_returns'] = np.log(1 + df['coinbase_returns'])

df.to_csv('btc_preprocessed1.csv',index = False)
df_plt = df['binance_btc_closing_price']


plt.plot(df['binance_btc_closing_price'])
#plt.plot(df['coinbase_btc_closing_price'])
plt.title('Bitcoin price on Binance')
plt.ylabel('Bitcoin price ($)')
plt.xlabel('Unix timestamps (s)')
# plt.legend(['Binance'])
plt.show()
#2) We want to break the data into 1440 subset

# data = df[['binance_btc_closing_price','coinbase_btc_closing_price','binance_log_returns','coinbase_log_returns']].to_numpy()
# #
# scaler = MinMaxScaler(feature_range=(0,1)).fit(data)
# normalized_data = scaler.transform(data)
#
# data_len = normalized_data.shape[0]
# new_data_len = data_len - data_len%1440
#
#
# normalized_data = normalized_data[:new_data_len,:] #We take of the extra data, with no full day at the end
# normalized_data = normalized_data.reshape(-1,1440,4)
# #print(data)
# print(data.shape)
#
#
# with open('btc.npy', 'wb') as f:
#     np.save(f,data)
