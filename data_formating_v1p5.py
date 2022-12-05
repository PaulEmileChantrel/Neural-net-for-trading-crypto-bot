import pandas as pd
import numpy as np
from tempfile import TemporaryFile
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from stockstats import wrap
#We format the data and add new features

#1) We want to erase the difference between the two exchanges price
# Since Binance give the BTC/USDT and Coinbase give the BTC/USD, we want to make sure that we dont have a small shift in all our data

df = pd.read_csv('btc_ohlcv_data.csv',index_col=[0])
#print(df)

df['diff'] = df['b_close'].astype('float32')-df['c_close'].astype('float32')
m = df['diff'].mean()#In average, the difference should be $0
print(m)

#We have m = -1.669 $ so we subtract it to coinbase price
df['c_open'] = df['c_open'] + m
df['c_high'] = df['c_high'] + m
df['c_low'] = df['c_low'] + m
df['c_close'] = df['c_close'] + m


bin_df = df[['b_open','b_high','b_low','b_close','b_volume']]
bin_df.reset_index(inplace=True)
bin_df.rename(columns = {'index':'Date','b_open':'Open','b_high':'High','b_low':'Low','b_close':'Close','b_volume':'Volume'},inplace=True)


coin_df = df[['c_open','c_high','c_low','c_close','c_volume']]

coin_df.reset_index(inplace=True)
coin_df.rename(columns = {'index':'Date','c_open':'Open','c_high':'High','c_low':'Low','c_close':'Close','c_volume':'Volume'},inplace=True)



bin_df = wrap(bin_df)
coin_df = wrap(coin_df)


df['b_macd'] = bin_df['macd']
df['c_macd'] = coin_df['macd']
df['b_rsi_7'] = bin_df['rsi_7']
df['c_rsi_7'] = coin_df['rsi_7']
df['b_rsi_21'] = bin_df['rsi_21']
df['c_rsi_21'] = coin_df['rsi_21']
df['b_sma_7'] = bin_df['close_7_sma']
df['c_sma_7'] = coin_df['close_7_sma']
df['b_sma_21'] = bin_df['close_21_sma']
df['c_sma_21'] = coin_df['close_21_sma']
df['b_wr_7'] = bin_df['wr_7']
df['c_wr_7'] = coin_df['wr_7']
df['b_wr_21'] = bin_df['wr_21']
df['c_wr_21'] = coin_df['wr_21']
df['b_log_ret'] = bin_df['log-ret']
df['c_log_ret'] = coin_df['log-ret']
bin_df['ups'], bin_df['downs'] = bin_df['change'] > 0, bin_df['change'] < 0
coin_df['ups'], coin_df['downs'] = coin_df['change'] > 0, coin_df['change'] < 0
df['b_ups_7_c'] = bin_df['ups_7_c']
df['c_ups_7_c'] = coin_df['ups_7_c']
df['b_ups_14_c'] = bin_df['ups_14_c']
df['c_ups_14_c'] = coin_df['ups_14_c']
df['b_downs_7_c'] = bin_df['downs_7_c']
df['c_downs_7_c'] = coin_df['downs_7_c']
df['b_downs_14_c'] = bin_df['downs_14_c']
df['c_downs_14_c'] = coin_df['downs_14_c']
print(df.head())
df.to_csv('btc_prossessed_V1p5.csv')
