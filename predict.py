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
import xgboost as xgb

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#%%
df = pd.read_csv('/home/apeppels/StockAnalysis/in_a_minute_ta.csv')

#%%
df.Timestamp = pd.to_datetime(df.Timestamp)

# Resampling to daily frequency
df.index = df.Timestamp
df.drop('Unnamed: 0',axis=1,inplace=True)
df.drop('Timestamp',axis=1,inplace=True)

#%%
def backtest(prices, actions, start_capital = 100., fee = 0.075 / 100, verbose=False):
    capital = start_capital
    position = 0
    bet = start_capital
    for i, price in enumerate(prices):
        if not price:
            continue
        action = actions[i]
        if action == 2 and capital > 0:
            if verbose:
                print("buy")
            position += bet - (bet * fee)
            capital -= bet
        elif action == 1 and position > 0:
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
def tradepoints(prices, lookahead=22, fee_pct=0.075 / 100, margin_pct = .0 / 100, verbose=False):
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
                indices[index] = 1
        #looking for buypoint
        else:
            cheaper_in_future = [price > p for p in nextprices] 
            worthwhile_buy = [price < p + fee for p in nextprices]
            if not True in cheaper_in_future and True in worthwhile_buy:
                lastval = price
                if verbose:
                    print('buy')
                up = True
                indices[index] = 2
        index += 1
    return indices

def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test

#%%
def predict(model, x):
    return model.predict(np.array(x).reshape((1,-1)))

#%%
def ta_data_to_sequences(data, how_much, offset):
    max_signal_searchback = 5000
    columns_per_category = 500
    resolution_offsets = {
            'lo': 50000,
            'mid': 5000,
            'hi': 500
            }

    res = []
    # how many rows do we want our matrix to have?
    for i in range(how_much):
        row = []
        # for each category, step through the price/volume data
        for resolution, timerange in resolution_offsets.items():
            steps = timerange / columns_per_category
            # for each data point, add it tot he feature vector
            for ta_data in data[-how_far_back - timerange - offset :-offset -i: steps]:
                row.append(ta_data['close'])
                row.append(ta_data['Amount'])
        res.append(row)
        
        # once we are done with the time series, append for each CDL
        # signal the time ago it was last spotted
        for signal in [key in data.keys() if 'CDL' in key]:
            # if we do not encounter it, store -1
            signal_offset = -1
            # look backwards to see if we encounter the signal anywhere in our history
            for i in range(max_signal_searchback):
                if data[signal][-offset - i] != 0:
                    # if we do, store offset i and look no further
                    signal_offset = i
                    break
            row.append(signal_offset)
        return res
        
#%%
df['change']=0
fwd_len = 30
#idx = np.arange(len(df),5484,-500)
#for index in idx:
cut = 300 + fwd_len

data = df[-cut:]
prices = data['close']
plot_x = data.index

#%%
labels = tradepoints(prices, lookahead=22)
#%%
# label the data, backtest and prepare plot
y1 = labels.values
y2 = prices
#df.at[x,'change'] = labels

backtest(y2, y1)
#%%
val_split = 10000 
x = data.values[:-val_split,2:]
y = labels[:-val_split]

x_val = data.values[-val_split:,2:]
y_val = labels[-val_split:]
#x_train, x_test, y_train, y_test = non_shuffling_train_test_split(x, y, test_size=0.2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

weight_ratio_sell = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 
1]))
weight_ratio_buy = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 
2]))
    
w_train = np.zeros(len(y_train))
w_train[y_train==2] = weight_ratio_buy
w_train[y_train==1] = weight_ratio_sell
w_train[y_train==0] = 1

w_test = np.zeros(len(y_test))
w_test[y_test==2] = weight_ratio_buy
w_test[y_test==1] = weight_ratio_sell
w_test[y_test==0] = 1
#%%
train_data = lightgbm.Dataset(x_train, label=y_train, weight=w_train)
test_data = lightgbm.Dataset(x_test, label=y_test, weight=w_test)

parameters = {
    'application': 'multiclass',
    'objective': 'multiclass',
    'num_classes': 3,
    'metric': 'multi_logloss',
    'boosting': 'gbdt',
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'learning_rate': 0.1,
    'verbose': 0
}

lgb_classifier = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=20)

#%%
exgb_classifier = xgb.XGBClassifier( verbosity=2)

#%%
exgb_classifier.fit(x_train,y_train,sample_weight=w_train)

#%%
# predict and plot
y_pred = exgb_classifier.predict(data.values[:,2:])

#%%
display = 200
y_pred = lgb_classifier.predict(data.values[-display:,2:])
y_pred = np.array([np.argmax(x) for x in y_pred])


#%%
y2 = data['close'][-display:]
y1 = y_pred
#y1 = y_val[-display:]
plot_x = data.index[-display:]
#%%
backtest(y2, y1)
#%%
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('change', color=color)
ax1.plot(plot_x, y1, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('BTC_USD', color=color)  # we already handled the x-label with ax1
ax2.plot(plot_x, y2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped


plt.show()
