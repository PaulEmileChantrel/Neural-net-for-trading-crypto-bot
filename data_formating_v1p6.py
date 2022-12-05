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

## Aggragate

def aggregate(df,bucket):
    new_df = pd.DataFrame(columns=['b_open','b_high','b_low','b_close','b_volume','c_open','c_high','c_low','c_close','c_volume'])
    minutes_index = df.index
    cols = new_df.columns
    for i in range(df.shape[0]):

        if i%bucket == 0:#new open
            idx = minutes_index[i]
            b_open = df['b_open'].iloc[i]
            b_high = df['b_high'].iloc[i]
            b_low = df['b_low'].iloc[i]
            b_volume = df['b_volume'].iloc[i]

            c_open = df['c_open'].iloc[i]
            c_high = df['c_high'].iloc[i]
            c_low = df['c_low'].iloc[i]
            c_volume = df['c_volume'].iloc[i]


        else:
            b_volume += df['b_volume'].iloc[i]
            c_volume += df['c_volume'].iloc[i]
            b_low = min(b_low,df['b_low'].iloc[i])
            c_low = min(c_low,df['c_low'].iloc[i])
            b_high = max(b_high,df['b_high'].iloc[i])
            c_high = max(c_high,df['c_high'].iloc[i])

        if i%(bucket-1) == 0 and i!=0:
            b_close = df['b_close'].iloc[i]
            c_close = df['c_close'].iloc[i]
            dict = {}
            row = [b_open,b_high,b_low, b_close, b_volume,c_open,c_high,c_low, c_close, c_volume]

            for k,j in enumerate(cols):
                dict[j] = row[k]
            new_df1 = pd.DataFrame(dict,index = [idx])
            new_df = pd.concat([new_df,new_df1],ignore_index=True)
    return new_df

def aggregateV2(df,bucket):
    cols = df.columns
    new_df = pd.DataFrame(columns=cols[:-1])
    minutes_index = df.index
    np_arr = df[cols[:-1]].to_numpy()
    new_arr = []
    for i in range(0,np_arr.shape[0]-bucket,bucket):
        idx = minutes_index[i]
        b_open = np_arr[i,0]
        b_high = np.max(np_arr[i:i+bucket,1])
        b_low = np.min(np_arr[i:i+bucket,2])
        b_close = np_arr[i+bucket-1,3]
        b_volume = np.sum(np_arr[i:i+bucket,4])

        c_open = np_arr[i,5]
        c_high = np.max(np_arr[i:i+bucket,6])
        c_low = np.min(np_arr[i:i+bucket,7])
        c_close = np_arr[i+bucket-1,8]
        c_volume = np.sum(np_arr[i:i+bucket,9])

        new_arr.append([idx,b_open,b_high,b_low,b_close,b_volume,c_open,c_high,c_low,c_close,c_volume])
    new_arr = np.array(new_arr)
    new_df1 = pd.DataFrame(new_arr[:,1:],columns = cols[:-1],index = new_arr[:,0])
    new_df = pd.concat([new_df,new_df1])
    return new_df

#df = aggregate(df,60)
df = aggregateV2(df,60)
#daily_df = aggregate(df,1440)

print(df)




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
df.to_csv('btc_prossessed_hourly_1p6.csv')
