#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:34:10 2019

@author: apeppels
"""

#continuation of oderbook_v2

import numpy as np
import matplotlib.pyplot as plt

orders = 200

window = 50

for offset in range(window, len(data), 3):
	f, ax = plt.subplots(3,1,figsize=(5,10))
	bp = [i[0] for i in data.depth.values[offset]['bids'][:orders]]
	bid_pr = [float(i[1]) for i in data.depth.values[offset]['bids'][:orders]]
	ap = [i[0] for i in data.depth.values[offset]['asks'][:orders]]
	ask_pr = [float(i[1]) for i in data.depth.values[offset]['asks'][:orders]]
	ax[0].plot(features.mids.iloc[offset - window: offset].values)
#	ax1 = ax[0].twinx()
	ax[1].bar(range(0,window),features.buy_volume.iloc[offset - window: offset].values, width=0.8, color='g',alpha=0.7)
	#ax[1].set_ylim((-5,5))
	ax[1].bar(range(0,window),-features.sell_volume.iloc[offset - window: offset].values, width=0.8, color='r',alpha=0.7)
#	plt.show()
	ax[2].plot(bp, np.cumsum(np.array(bid_pr)))
	ax[2].plot(ap, np.cumsum(np.array(ask_pr)))
	plt.show()

   #%%

from sklearn.linear_model import LinearRegression
X = features[['best_bid','best_ask','spread','volume',]]

reg = LinearRegression().fit(X, y)
 


#%%


tmp = np.arange(1,7).reshape((3,1,2))
print(tmp.ravel().shape)
print(np.squeeze(tmp).shape)
print(tmp.flatten().shape)