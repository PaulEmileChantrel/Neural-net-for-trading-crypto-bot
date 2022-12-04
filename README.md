# Neural Network for trading bot


The goal of this project is to train a neural net using reinforcement machine learning and Deep-Q learning to trad bitcoin.
We will give the NN the Bitcoin price from Binance and Coinbase at 1 min interval.



<p align='center'>
<img width='720px' alt='bitcoin price for neural net training' src='https://user-images.githubusercontent.com/96018383/205213004-111e3147-0e0d-47b6-be9f-8cf85ddf8623.png'>
</p>

We want to see wether the bot will do arbitrage between the two or simply trade in and out of Bitcoin.

## Neural Net V1

For the first version of the neural net, we use a simple LSTM/GRU neural net to predict the next-day value.
(With the help of https://www.youtube.com/watch?v=dKBKNOn3gCE.)

We add a feature called log returns such as $r = log(1+ current-price/previous-price)$. Without this feature, our model will not be able to predict values outside the range he was trained on.

We split the set into a training set and a testing set as we can see in the figure below:
<p align='center'>
<img width='720px' alt='bitcoin price for neural net training' src='https://user-images.githubusercontent.com/96018383/205468397-5fef6402-44f4-4d68-8a4c-b2aa7a250796.png'>
</p>


Here is a price prediction of the next price using the 5 previous prices in the dataset :
<p align='center'>
<img width='720px' alt='bitcoin price for neural net training' src='https://user-images.githubusercontent.com/96018383/205469710-3c537893-af24-47d8-8225-117f4aeb9093.png'>
</p>

It may look like our NN is good at predicting the market but if we create a simple trading bot that uses the NN prediction we can quickly realize that the bot doesn't make any money.
We use the following trading rules : 
* If you own cash, all in if we predict tomorrow will be up
* If you own cash, do nothing if we predict tomorrow will be down
* If you own BTC, sell all if we predict tomorrow is down OR if today was down and we hit the stop loss (10% loss)
* If you own BTC, do nothing if we predict tomorrow will be up

We can predict a correct up or down result about 50% of the time (sometimes above, sometimes below) and the average variation is the same for a correct or incorrect prediction. In practice, the trading bot will sometimes earn a few % and sometimes lose 10% (when the stop loss gets hit).
Conclusion: this bot is not better than random.

We can try to add more features for the training like rsi, macd, ema, ma, adx, etc. 
It is also possible that the data look more random on a minute time scale than an hourly or daily one where the market sentiment would have more impact and could be easier to read.

### Neural Net with more features

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




