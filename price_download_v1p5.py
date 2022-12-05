#Download Prices from Binance and Coinbase and put them on a .csv file
# This time, we keep the date,O,H,L,C,V data

import requests
import time
import pandas as pd
import numpy as np

now = int(time.time()*1000)
now = now - now%(60*1000)#rounding to the lowest min
b_close = []
b_open = []
b_high = []
b_low = []
b_volume = []
binance_timestamps = []
c_close = []
c_open = []
c_high = []
c_low = []
c_volume = []

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
        b_open.append(response[j][1])
        b_high.append(response[j][2])
        b_low.append(response[j][3])
        b_close.append(response[j][4])
        b_volume.append(response[j][5])
    if i%(data_len//10)==0:
        print(f'Binance advancement : {i*100/data_len:0.2f} %')
binance_timestamps = list(map(int, binance_timestamps))
#print(b_close)
print(len(b_close))
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
        c_open.append(response[j][1])
        c_high.append(response[j][2])
        c_low.append(response[j][3])
        c_close.append(response[j][4])
        c_volume.append(response[j][5])
    if i%(c_data_len//10)==0:
        print(f'Coinbase advancement : {i*100/c_data_len:0.2f} %')

print(len(c_close),len(coinbase_timestamps),len(c_open))
bin_df = pd.DataFrame({'b_open':b_open,'b_high':b_high,'b_low':b_low,'b_close':b_close,'b_volume':b_volume}, index = binance_timestamps)
print(bin_df.head())
cb_df = pd.DataFrame({'c_open':c_open,'c_high':c_high,'c_low':c_low,'c_close':c_close,'c_volume':c_volume}, index = coinbase_timestamps)
print(cb_df.head())

df = bin_df.join(cb_df)

df.fillna('',inplace=True)
df = df[df['c_close']!='']
df = df.sort_index(ascending=True)
df.to_csv('btc_ohlcv_data.csv')
# df['diff'] = df['b_close'].astype('float32')-df['c_close'].astype('float32')
# print(df['diff'].mean())
print(len(c_close))
print(c_data_len)
print(f'Total time : {time.time()-now/1000}')
