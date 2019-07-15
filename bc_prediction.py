#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:46:28 2019

@author: lisatostrams
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime
import os
warnings.filterwarnings('ignore')
#plt.style.use('seaborn-poster')

f =os.listdir('bitcoin-historical-data')
f = [file for file in f if 'csv' in file]
df = pd.read_csv('bitcoin-historical-data/'+f[1],header=None,names=['Timestamp','Price','Amount'])

df.Timestamp = pd.to_datetime(df.Timestamp, unit='s')

# Resampling to daily frequency
df.index = df.Timestamp
ohlc = df['Price'].resample('60S').ohlc()
ohlcpa = ohlc.join(df[['Price','Amount']].resample('60S').mean())

df.head()
df.tail()

ohlcpa['Volume'] = ohlcpa['Price'] * ohlcpa['Amount']
ohlcpa.dropna(axis=0,inplace=True)

# next we need to fix the OHLC (open high low close) data which is a continuous timeseries so
# lets fill forwards those values...
#df = df.resample('').mean()
#
## Resampling to monthly frequency
#df_month = df.resample('M').mean()
#
## Resampling to annual frequency
#df_year = df.resample('A-DEC').mean()


#df['iam'] = df['Price'].shift(periods=1)
#
df['Price'].plot()
#plt.plot(df['Close'])
#plt.plot(df['iam'])

#ohlcpa.to_csv('in_a_minute.csv')
