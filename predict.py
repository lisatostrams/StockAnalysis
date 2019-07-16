#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 21:51:29 2019

@author: apeppels
"""

import sys
import numpy as np
import pandas as pd
import lightgbm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%%
df = pd.read_csv('in_a_minute_ta.csv')

#%%
df.Timestamp = pd.to_datetime(df.Timestamp)

# Resampling to daily frequency
df.index = df.Timestamp
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('Timestamp',axis=1,inplace=True)

#%%
def backtest(prices, actions, start_capital = 10., fee = 0.3 / 100, verbose=False):
    capital = start_capital
    position = 0
    bet = start_capital
    for i, price in enumerate(prices):
        action = actions[i]
        if action == 1 and capital > 0:
            if verbose:
                print("buy")
            position += bet - (bet * fee)
            capital -= bet
        elif action == -1 and position > 0:
            if verbose:
                print("sell")
            capital += bet - (bet * fee)
            position -= bet
        # propagate price changes
        change_factor = (prices[i] - prices[i-1]) / prices[i]
        position += position * change_factor
        if verbose:
            print(position, capital, change_factor)
    portfolio = position + capital
    earning_pct = round(((float(portfolio) / start_capital ) -1 ) *100, 2)
    print("Earnings over {} points: {} percent".format(len(prices), earning_pct))
    print(portfolio)
        
#%%
def tradepoints(prices, lookahead=22, fee_pct=0.075 / 100, margin_pct = .3 / 100, verbose=False):
    indices = prices.apply(lambda x: 0)
    up = False
    index = 0
    lastval = prices[0]
    if verbose:
        print(len(prices))
    while index < len(prices) - lookahead:
        price = prices[index]
        nextprices = prices[index:index+lookahead]
        fee = price * (fee_pct + margin_pct)
        if verbose:
            print(lastval, price, fee)
        label = 0
        # looking for sellpoint
        if up:
            worth_more_in_future = [price < p for p in nextprices]
            worthwhile_sell = price > lastval + fee
            if not True in worth_more_in_future and worthwhile_sell:
                lastval = price
                if verbose:
                    print('sell')
                up = False
                indices[index] = -1
        #looking for buypoint
        else:
            cheaper_in_future = [price > p for p in nextprices] 
            worthwhile_buy = [price < p + fee for p in nextprices]
            if not True in cheaper_in_future and True in worthwhile_buy:
                lastval = price
                if verbose:
                    print('buy')
                up = True
                indices[index] = 1
        index += 1
    return indices

#%%
df['change']=0
fwd_len = 30
#idx = np.arange(len(df),5484,-500)
#for index in idx:
cut = 500000 + fwd_len
prices = df['close'][-cut:-fwd_len]

labels = tradepoints(prices, lookahead=22)
#%%

x = data.iloc[:, 1:]
y = labels

x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

#%%

train_data = lightgbm.Dataset(x, label=y)
test_data = lightgbm.Dataset(x_test, label=y_test)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)
#
# Create a submission
#%%
x = labels.index
y1 = labels.values
y2 = prices
df.at[x,'change'] = labels

backtest(y2, y1)

#%%
%matplotlib inline

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('change', color=color)
ax1.plot(x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('BTC_USD', color=color)  # we already handled the x-label with ax1
ax2.plot(x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.show()
