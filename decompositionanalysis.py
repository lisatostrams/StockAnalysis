#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:19:13 2019

@author: lisatostrams
"""
import pandas as pd
import numpy as np
minutes = pd.read_csv("bitcoin-historical-data/in_a_minute.csv")
N = len(minutes)
print('There are {} minutes in the file.'.format(N))
cols = list(minutes.columns.values)
minutes.index = minutes.Timestamp
cols.remove('Timestamp')

X = minutes[cols]
y = minutes['Price'].shift(periods=60)
idx = minutes['Timestamp']
del minutes
X=X.replace([np.inf, -np.inf], np.nan)
X=X.fillna(0)
y=y.fillna(0)
#%%

import matplotlib.pyplot as plt 
X.Price.plot()


X['MAday']= X.Price.rolling(window=1440).mean()
X['Mminday'] = X.Price.rolling(window=1440).min()
X['Mmaxday'] = X.Price.rolling(window=1440).max()


def plot(X,time=0):
    cols = list(X.columns.values)
    cols.remove('Volume')
    cols.remove('Amount')
    if time>0:
        X.loc[idx[-time:]][cols].plot()
    else:
        X[cols].plot()
#%%
from scipy.fftpack import fft, ifft
def fourier_features(signal,nbins=10):
    n = 15000
    ps = abs(ifft(signal))
    ps = ps[1:(n+1)]
    
    time_step = 1 / 4000000
    f={}
    f['fft_power_sum']= np.trapz(ps, dx=time_step*n)
    f['fft_power_max']=max(ps)
    f['fft_power_argmax'] = np.argmax(ps)*time_step
    bins = np.arange(0,n+1,step=n//nbins)
    for b in range(0,nbins):  
        power = np.trapz(ps[bins[b]:bins[b+1]],dx=time_step*n)
        f['fft_power_bin{}'.format(b)] = power
    return f