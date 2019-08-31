 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 23:47:42 2019

@author: apeppels
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy

data =  open("BTC_USDT2.json").read().splitlines()

deltas = []
trades = []
for x in data:
	try:
		a = json.loads(x.replace("u'", '"').replace("'", '"').replace("True", "true").replace("False","false")) 
		if a["stream"] == "btcusdt@depth":
			deltas.append(a["data"])
		if a["stream"] == "btcusdt@aggTrade" and not len(trades) :
			trades.append(a["data"])
		
	except Exception as e:
		print(e)

depth_df = pd.DataFrame(deltas)
trades_df = pd.DataFrame(trades)
trades_df['timestamp'] = pd.to_datetime(trades_df['E'], unit='ms')
depth_df['timestamp'] = pd.to_datetime(depth_df['E'], unit='ms')
#depth_df.set_index("timestamp")
depth_df.index = depth_df.timestamp
trades_df.index = trades_df.timestamp
depth_df = depth_df.rename(columns = {"a": "ask", "b":"bid"})


#%%

def extract_book_features(book):
    if len(book["ask"]) == 0 or len(book["bid"]) == 0:
        return 0,0,0,0
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


seconds = pd.date_range(start=depth_df['timestamp'].min(), end=depth_df['timestamp'].max(), freq='s')
#books = pd.DataFrame(columns=['ask','bid'],index=seconds)
books = []
book_features = []

for s in seconds:
	orderbook_mask = ((depth_df["timestamp"] > s) & (depth_df["timestamp"] < s + pd.Timedelta(1, unit='s')))
	trades_mask = ((trades_df["timestamp"] > s) & (trades_df["timestamp"] < s + pd.Timedelta(1, unit='s')))
	deltas_in_second = depth_df.loc[orderbook_mask]
	trades_in_second = trades_df.loc[trades_mask]
	
	for side, updates in deltas_in_second.loc[:, ["ask", "bid"]].iteritems():
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
	book["trades"] = trades_in_second
	book_features.append(extract_book_features(book))
	books.append(book)
	
books = pd.DataFrame(books)
books.set_index("ts")
books.to_csv("orderbooks_btc_usdt2.csv")

#%%
def plot_n(y):
    x = np.arange(0,len(y),1)
    for i in range(len(y[0])):
        plt.plot(x,[z[i] for z in y])
    plt.show()
    
def plot_1(y):
    x = np.arange(0,len(y),1)
    plt.plot(x,y)
    plt.show()

#%%
offset = 0
amount = 10000
# those 2s are to work around a missing datapoint
plot_1([x[0] for x in book_features[2+offset:2 + offset + amount]])

#plot_1([x[3] for x in book_features[2+offset:2 + offset + amount]])
min_max_orders = [(x[1], x[2]) for x in book_features[5+offset:5 + offset + amount]]
plot_n(min_max_orders)
#%%

distances = []

for i, x in enumerate(min_max_orders):
	j = 0
	found = False
	while not found and j < 1000 and j + i < len(min_max_orders):
		future_price = min_max_orders[i + j][1] 
		fee = (future_price * 0.001) + (x[0] * 0.001)
		
		#fee = 0
		diff = future_price - x[0] - (fee * 1.3)
		j += 1
		if diff > 0:
			found = True
			#print(diff)
	distances.append(j)

plot_1(distances)

