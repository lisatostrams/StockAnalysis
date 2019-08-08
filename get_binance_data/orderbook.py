#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:47:42 2019

@author: apeppels
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
import copy
data =  open("LINK_USDT_notick.json").read().splitlines()

deltas = []
for x in data:
	try:
		a = json.loads(x.replace("u'", '"').replace("'", '"').replace("True", "true").replace("False","false")) 
		if a["stream"] == "linkusdt@depth":
			deltas.append(a["data"])
	except Exception as e:
		print(e)

df = pd.DataFrame(deltas)
df['timestamp'] = pd.to_datetime(df['E'], unit='ms')
#df.set_index("timestamp")
df.index = df.timestamp
df = df.rename(columns = {"a": "ask", "b":"bid"})

trades = []

#%%

def extract_book_features(book):
    if len(book["ask"]) == 0 and len(book["bid"]) == 0:
        return 0,0
    asks = sorted(book["ask"].items())
    bids = sorted(book["bid"].items(), reverse=True)
    min_ask = asks[0][0]
    max_bid = bids[0][0]
    distance = min_ask - max_bid
    midprice = (min_ask + max_bid) / 2
    
    
    return distance, min_ask, max_bid, midprice
#%%
orderbook = {
		"ask": {},
		"bid": {}
}


seconds = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='s')
#books = pd.DataFrame(columns=['ask','bid'],index=seconds)
books = []
book_features = []

for s in seconds:
	mask = ((df["timestamp"] > s) & (df["timestamp"] < s + pd.Timedelta(1, unit='s')))
	deltas = df.loc[mask]	
	for side, updates in deltas.loc[:, ["ask", "bid"]].iteritems():
		if len(updates) != 0:	
			print(side,updates)
			for price, update in updates.iloc[0]:
				price = float(price)
				update = float(update)
				if float(update) != 0:
					orderbook[side][price] = update
				elif price in orderbook[side]:
					orderbook[side].pop(price)
		print(len(orderbook["ask"]), len(orderbook["bid"]))
	book = copy.deepcopy(orderbook)
	book["ts"] = s
	book_features.append(extract_book_features(book))
	books.append(book)
	
books = pd.DataFrame(books)
books.set_index("ts")
books.to_csv("orderbooks_link_usdt.csv")

#%%
def plot_2(y):
    x = np.arange(0,len(y),1)
    plt.plot(x,[z[0] for z in y])
    plt.plot(x,[z[1] for z in y])
    plt.show()
    
def plot_1(y):
    x = np.arange(0,len(y),1)
    plt.plot(x,y)
    plt.show()

#%%
plot_1([x[0] for x in book_features[2:1000]])

plot_1([x[3] for x in book_features[2:1000]])
plot_2([(x[1], x[2]) for x in book_features[2:1000]])
#%%
