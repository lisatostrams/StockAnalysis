#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 13:17:50 2019

@author: lisatostrams
"""

import sys
import numpy as np
import tushare as ts
import pandas as pd
import talib as tb
from xgboost import XGBClassifier
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score


# define pivot variables for easy use
def technical(df):
    open = df['open'].values
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['Volume'].values
    # define the technical analysis matrix
    retn = np.array([
        tb.MA(close, timeperiod=5),                                         # 1
        tb.MA(close, timeperiod=10),                                        # 2
        tb.MA(close, timeperiod=20),                                        # 3
        tb.MA(close, timeperiod=60),                                        # 4
        tb.MA(close, timeperiod=90),                                        # 5
        tb.MA(close, timeperiod=120),                                       # 6

        tb.ADX(high, low, close, timeperiod=20),                            # 7
        tb.ADXR(high, low, close, timeperiod=20),                           # 8

        tb.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],    # 9
        tb.RSI(close, timeperiod=14),                                       # 10

        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0],  # 11
        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1],  # 12
        tb.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2],  # 13

        tb.AD(high, low, close, volume),                                    # 14
        tb.ATR(high, low, close, timeperiod=14),                            # 15

        tb.HT_DCPERIOD(close),                                              # 16

        tb.CDL2CROWS(open, high, low, close),                               # 17
        tb.CDL3BLACKCROWS(open, high, low, close),                          # 18
        tb.CDL3INSIDE(open, high, low, close),                              # 19
        tb.CDL3LINESTRIKE(open, high, low, close),                          # 20
        tb.CDL3OUTSIDE(open, high, low, close),                             # 21
        tb.CDL3STARSINSOUTH(open, high, low, close),                        # 22
        tb.CDL3WHITESOLDIERS(open, high, low, close),                       # 23
        tb.CDLABANDONEDBABY(open, high, low, close, penetration=0),         # 24
        tb.CDLADVANCEBLOCK(open, high, low, close),                         # 25
        tb.CDLBELTHOLD(open, high, low, close),                             # 26
        tb.CDLBREAKAWAY(open, high, low, close),                            # 27
        tb.CDLCLOSINGMARUBOZU(open, high, low, close),                      # 28
        tb.CDLCONCEALBABYSWALL(open, high, low, close),                     # 29
        tb.CDLCOUNTERATTACK(open, high, low, close),                        # 30
        tb.CDLDARKCLOUDCOVER(open, high, low, close, penetration=0),        # 31
        tb.CDLDOJI(open, high, low, close),                                 # 32
        tb.CDLDOJISTAR(open, high, low, close),                             # 33
        tb.CDLDRAGONFLYDOJI(open, high, low, close),                        # 34
        tb.CDLENGULFING(open, high, low, close),                            # 35
        tb.CDLEVENINGDOJISTAR(open, high, low, close, penetration=0),       # 36
        tb.CDLEVENINGSTAR(open, high, low, close, penetration=0),           # 37
        tb.CDLGAPSIDESIDEWHITE(open, high, low, close),                     # 38
        tb.CDLGRAVESTONEDOJI(open, high, low, close),                       # 39
        tb.CDLHAMMER(open, high, low, close),                               # 40
        tb.CDLHANGINGMAN(open, high, low, close),                           # 41
        tb.CDLHARAMI(open, high, low, close),                               # 42
        tb.CDLHARAMICROSS(open, high, low, close),                          # 43
        tb.CDLHIGHWAVE(open, high, low, close),                             # 44
        tb.CDLHIKKAKE(open, high, low, close),                              # 45
        tb.CDLHIKKAKEMOD(open, high, low, close),                           # 46
        tb.CDLHOMINGPIGEON(open, high, low, close),                         # 47
        tb.CDLIDENTICAL3CROWS(open, high, low, close),                      # 48
        tb.CDLINNECK(open, high, low, close),                               # 49
        tb.CDLINVERTEDHAMMER(open, high, low, close),                       # 50
        tb.CDLKICKING(open, high, low, close),                              # 51
        tb.CDLKICKINGBYLENGTH(open, high, low, close),                      # 52
        tb.CDLLADDERBOTTOM(open, high, low, close),                         # 53
        tb.CDLLONGLEGGEDDOJI(open, high, low, close),                       # 54
        tb.CDLLONGLINE(open, high, low, close),                             # 55
        tb.CDLMARUBOZU(open, high, low, close),                             # 56
        tb.CDLMATCHINGLOW(open, high, low, close),                          # 57
        tb.CDLMATHOLD(open, high, low, close, penetration=0),               # 58
        tb.CDLMORNINGDOJISTAR(open, high, low, close, penetration=0),       # 59
        tb.CDLMORNINGSTAR(open, high, low, close, penetration=0),           # 60
        tb.CDLONNECK(open, high, low, close),                               # 61
        tb.CDLPIERCING(open, high, low, close),                             # 62
        tb.CDLRICKSHAWMAN(open, high, low, close),                          # 63
        tb.CDLRISEFALL3METHODS(open, high, low, close),                     # 64
        tb.CDLSEPARATINGLINES(open, high, low, close),                      # 65
        tb.CDLSHOOTINGSTAR(open, high, low, close),                         # 66
        tb.CDLSHORTLINE(open, high, low, close),                            # 67
        tb.CDLSPINNINGTOP(open, high, low, close),                          # 68
        tb.CDLSTALLEDPATTERN(open, high, low, close),                       # 69
        tb.CDLSTICKSANDWICH(open, high, low, close),                        # 70
        tb.CDLTAKURI(open, high, low, close),                               # 71
        tb.CDLTASUKIGAP(open, high, low, close),                            # 72
        tb.CDLTHRUSTING(open, high, low, close),                            # 73
        tb.CDLTRISTAR(open, high, low, close),                              # 74
        tb.CDLUNIQUE3RIVER(open, high, low, close),                         # 75
        tb.CDLUPSIDEGAP2CROWS(open, high, low, close),                      # 76
        tb.CDLXSIDEGAP3METHODS(open, high, low, close)                      # 77
    ]).T
    return retn

data = pd.read_csv('in_a_minute.csv')
ta = technical(data)

#%%
with open('cols.txt') as f:
    cols = f.readlines()
cols = [c for c in cols if len(c)>1]
#%%
tadf = pd.DataFrame(ta,index = data.index,columns=cols)
data = data.join(tadf)
#%%
data.to_csv('in_a_minute_ta.csv')
#%%










