#Download Prices from Binance and Coinbase and put them on a .csv file

import requests
import time
import pandas as pd
import numpy as np

now = int(time.time()*1000)
now = now - now%(60*1000)#rounding to the lowest min
binance_btc_closing_price = []
binance_timestamps = []
coinbase_btc_closing_price = []
coinbase_timestamps = []

binance_main_url = 'https://api.binance.com'
binance_start_time = now-60*1000*1000
coinbase_start_time = now/1000-60*300
binance_end_time = now
coinbase_end_time = now/1000

data_len = 2880

for i in range(data_len):
    binance_url = binance_main_url+ '/api/v3/klines?symbol=BTCUSDT&interval=1m&startTime='+str(binance_start_time)+'&endTime='+str(binance_end_time)+'&limit=1000'
    binance_end_time = binance_start_time -1
    binance_start_time = binance_end_time - 60*1000*1000
    response = requests.get(binance_url)
    response = response.json()

    if len(response) < 100:
        print(response,i)
        break

    for j in range(len(response)):
        binance_timestamps.append(response[j][0]/1000)
        binance_btc_closing_price.append(response[j][4])#We only take the close price
    if i%(data_len//10)==0:
        print(f'Binance advancement : {i*100/data_len:0.2f} %')
binance_timestamps = list(map(int, binance_timestamps))
#print(binance_btc_closing_price)
print(len(binance_btc_closing_price))
print(f'Binance time : {time.time()-now/1000}')

coinbase_main_url = 'https://api.exchange.coinbase.com'
c_data_len = int(data_len*1000/300+1)
for i in range(c_data_len):
    coinbase_url = coinbase_main_url + '/products/BTC-USD/candles?start='+str(int(coinbase_start_time))+'&end='+str(int(coinbase_end_time))+'&granularity=60'
    coinbase_end_time = coinbase_start_time-1
    coinbase_start_time = coinbase_end_time - 60*300


    response = requests.get(coinbase_url)
    response = response.json()

    if len(response) < 10:
            print(response,len(response),i)
            break

    for j in range(len(response)):
        coinbase_timestamps.append(response[j][0])
        coinbase_btc_closing_price.append(response[j][4])#We only take the close price
    if i%(c_data_len//10)==0:
        print(f'Coinbase advancement : {i*100/c_data_len:0.2f} %')



bin_df = pd.DataFrame(binance_btc_closing_price, index = binance_timestamps,columns = ['binance_btc_closing_price'])

cb_df = pd.DataFrame(coinbase_btc_closing_price, index = coinbase_timestamps,columns = ['coinbase_btc_closing_price'])

# bin_df = bin_df.sort_values(by='binance_timestamps', ascending=False,ignore_index = True)
# cb_df = cb_df.sort_values(by='coinbase_timestamps', ascending=False,ignore_index = True)
#cb_df = cb_df.drop(0)
df = bin_df.join(cb_df)

df.fillna('',inplace=True)
df = df[df['coinbase_btc_closing_price']!='']
df = df.sort_index(ascending=True)
df.to_csv('btc_data.csv')
# df['diff'] = df['binance_btc_closing_price'].astype('float32')-df['coinbase_btc_closing_price'].astype('float32')
# print(df['diff'].mean())
print(len(coinbase_btc_closing_price))
print(c_data_len)
print(f'Total time : {time.time()-now/1000}')
