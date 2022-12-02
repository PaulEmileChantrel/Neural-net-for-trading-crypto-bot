# Neural Network for trading bot

The goal of this project is to train a neural net using reinforcement machine learning and Deep-Q learning to trad bitcoin.
We will give the NN the Bitcoin price from Binance and Coinbase at 1 min interval.



<p align='center'>
<img width='720px' alt='bitcoin price for neural net training' src='https://user-images.githubusercontent.com/96018383/205213004-111e3147-0e0d-47b6-be9f-8cf85ddf8623.png'>
</p>

We want to see wether the bot will do arbitrage between the two or simply trade in and out of Bitcoin.

## Neural Net V1

For the first version of the neural net, we use a simple LSTM neural net to predict the next day value.
(With the help of https://www.youtube.com/watch?v=dKBKNOn3gCE.)

## Neural Net V2

For the second version of the neural net, we use a reinforcement machine learning to teach the bot how to trade.

### Action Space :
6 possible actions

Buy y1 BTC on Binance -> 0

Sell y2 BTC on Binance -> 1

Do nothing on Biance -> 2

Buy y3 BTC on Coinbase -> 3

Sell y4 BTC on Coinbase -> 4

Do nothing on Coinbase

Actions on 1 exchange are discreate (you can't buy and sell at the same time on the same exchange).

y1 and y3 varies between 0 and the max num of possible btc buy with the current amount max_num = cash/btc_price

y2 and y4 varies between 0 and the number of btc

### Observations space :
Number of $ (cash = 100000 at t=0)

Number of BTC (btc = 0 at t=0)

(We considere that $ and BTC can move freely and instantly between Binance and Coinbase for now)


### Rewards :
If you earn $x after at the end of a min 1*x

If you loose $x after at the end of a min -1.5*x

If you make a trade of size y -0.001*y


### Episode Termination
Portfolio goes bellow $10,000 (-90%)

End of day




