#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:09:16 2019

@author: lisatostrams
"""

import numpy as np 
import pandas as pd 

import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')

#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):    
    return datetime.fromtimestamp(float(time_in_secs))

print('Data listing...')
print(os.listdir('bitcoin-historical-data'))
f =os.listdir('bitcoin-historical-data')

# read in the data and apply our conversion function, this spits out a DataFrame with the DateTimeIndex already in place
print('Using bitstampUSD_1-min_data...')
#data = pd.read_csv('bitcoin-historical-data/'+f[0], parse_dates=True, date_parser=dateparse, index_col=[0])

print('Total null open prices: %s' % data['Open'].isnull().sum())

data['Volume_(BTC)'].fillna(value=0, inplace=True)
data['Volume_(Currency)'].fillna(value=0, inplace=True)
data['Weighted_Price'].fillna(value=0, inplace=True)

# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so
# lets fill forwards those values...
data['Open'].fillna(method='ffill', inplace=True)
data['High'].fillna(method='ffill', inplace=True)
data['Low'].fillna(method='ffill', inplace=True)
data['Close'].fillna(method='ffill', inplace=True)

# check how we are looking now, should be nice and clean...
print(data)

# The first thing we need are our trading signals. The Turtle strategy was based on daily data and
# they used to enter breakouts (new higher highs or new lower lows) in the 22-60 day range roughly.
# We are dealing with minute bars here so a 22 minute new high isn't much to get excited about. Lets
# pick an equivalent to 60 days then. They also only considered Close price so lets do the same...

signal_lookback = 60 * 24 * 60 # days * hours * minutes

# here's our signal columns
data['Buy'] = np.zeros(len(data))
data['Sell'] = np.zeros(len(data))

# this is our 'working out', you could collapse these into the .loc call later on and save memory 
# but I've left them in for debug purposes, makes it easier to see what is going on
data['RollingMax'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).max()
data['RollingMin'] = data['Close'].shift(1).rolling(signal_lookback, min_periods=signal_lookback).min()
data.loc[data['RollingMax'] < data['Close'], 'Buy'] = 1
data.loc[data['RollingMin'] > data['Close'], 'Sell'] = -1

#%% lets now take a look and see if its doing something sensible


fig,ax1 = plt.subplots(1,1,figsize=(8,10))
ax1.plot(data['Close'])
y = ax1.get_ylim()
ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])

ax2 = ax1.twinx()
ax2.set_position(matplotlib.transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
ax2.plot(data['Buy'], color='#77dd77')
ax2.plot(data['Sell'], color='#dd4444')
plt.show()
