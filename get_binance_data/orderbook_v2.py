#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from json import loads
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#%%
fname = 'btc'

plt.rcParams["figure.figsize"] = [8,3]

def plot_1(y):
    x = np.arange(0,len(y),1)
    plt.plot(x,y)
    plt.show()
	
def bin_orders(row):
    b = row['depth']
    sides = ['bids', 'asks']
    books = {k:[] for k in sides}
    bin_size = 100
    for k in sides:
        is_bids = k == 'bids'
        orders = 1
        b[k] = sorted(b[k], reverse=is_bids)
        #print(b[k][-10:])
        for dist_idx, samples in enumerate([int(100 / int((i**1.9))) for i in range(1,11)]):
            dist_idx = dist_idx + 1
            distance = bin_size // samples
            while i and orders < dist_idx * bin_size and orders + distance < len(b[k]):
                avgprice = (float(b[k][orders][0]) + float(b[k][orders + distance][0]) ) / 2
                print(avgprice)
                #print(float(b[k][orders][0]), float(b[k][orders +distance][0]))
                #print(orders, distance, orders+distance, b[k][904], b[k][orders:orders+distance])
                books[k].append([avgprice, sum([float(x[1]) for x in b[k][orders:orders+distance]])]) 
                orders += distance
            orders = dist_idx * bin_size
    row['depth'] = books
    return row

def bin_orders(row):
    b = row['depth']
    sides = ['bids', 'asks']
    books = {k:[] for k in sides}
    for k in sides:
        rev = k == 'asks'
        #b[k] = sorted(b[k], reverse=rev, key=lambda tup: tup[0])
        orders = 0
        resolution = 1
        while orders + resolution < len(b[k]):
            avgprice = (float(b[k][orders][0]) + float(b[k][orders + resolution][0]) ) / 2
            books[k].append([avgprice, sum([float(x[1]) for x in b[k][orders:orders+resolution]])])
            orders += resolution
            resolution = orders // 30 + 1      
    row['depth'] = books
    return row

#%%				
data = []
i=0
print('parsing lines')
for filename in ["BTCUSDT_marketdata.json", "BTCUSDT_marketdata.json.1"]:
	with open(filename) as f:	 
		line = f.readline()
		while line:
			i += 1
			print("\r{}".format(i), end='\t\t')
			line = f.readline()
			#if not i % 5 == 0:
			#	continue
    
			try:
				x = loads(line)
				data.append(bin_orders(x))
			except Exception as e:
				print(e)
#%%
print('making dataframe')
data = pd.DataFrame(data)
data['timestamp'] = pd.to_datetime(data['time'], unit='s')
data = data.set_index(data.timestamp)
plot_1([x['asks'][0][0] for x in data.depth.values])
#data.to_csv('btc')
#%%
#data = pd.read_csv('btc', index_col=0)[['depth','trades']].applymap(lambda x: eval(x))
#data[['depth','trades']] = data[['depth','trades']].applymap(lambda x: eval(x))
#%%

print("calculating features")
features = pd.DataFrame(index=data.index)
features['best_bid'] = data.depth.apply(lambda x: float(x['bids'][0][0]))
features['best_ask'] = data.depth.apply(lambda x: float(x['asks'][0][0]))
features['mids'] = (features.best_ask + features.best_bid) / 2
features['spread'] = features.best_ask-features.best_bid
features['volume'] = data.trades.apply(lambda x: sum([float(y["q"]) for y in x]))
features['sell_volume'] = data.trades.apply(lambda x: sum([float(y["q"]) for y in x if y["sell"]]))
features['buy_volume'] = data.trades.apply(lambda x: sum([float(y["q"]) for y in x if not y["sell"]]))

price_distance = 20
close_asks = data.depth.apply(lambda x: [y for y in x['asks'] if float(y[0]) - float(x['asks'][0][0]) < price_distance])
#%%
def flatten_orderbook(row, order_limit=1000, trade_limit=20):
	book = row.depth
	midprice = (float(book['asks'][0][0]) + float(book['bids'][0][0])) / 2
	orders = np.array([midprice],dtype=np.float64)
	for k in ['bids', 'asks']:
		orders = np.append(orders, np.array([(float(x[0]) - midprice, float(x[1])) for x in book[k][:order_limit]],dtype=np.float64).ravel())
	trades = np.array([(trade['p'], trade['q'], int(trade['sell'])) for trade in row.trades],dtype=np.float64).ravel()[:trade_limit]
	return np.append(orders, np.pad(trades, (0, trade_limit*3 - len(trades)), mode='constant'))		
#%%
print('constructing X,y')	
scaler = MinMaxScaler()
X = data.apply(flatten_orderbook, axis=1)
y = data.depth.apply(lambda x: float(x['bids'][0][0]))

#%%
X_uncut = pd.DataFrame(scaler.fit_transform(X.values.tolist()), index=X.index)
X_uncut.columns = X_uncut.columns.astype(str)

window_size = 100
for i in range(window_size):
    X_uncut["prev_{}".format(i)] = X_uncut["0"].shift(periods=-int(i*1.5))
    
X_uncut.to_csv('btc_X')
y.to_csv('btc_y', header=['price'])
#%%
X_uncut = pd.read_csv('btc_X').set_index('timestamp')
#%%
lookahead = 100
X = X_uncut[:-lookahead]
prices = pd.read_csv('btc_y').set_index('timestamp')
#prices = data.depth.apply(lambda x: float(x['bids'][0][0]))
y = prices.shift(periods=-lookahead) - prices
y = y[:-lookahead]
prices = prices[:-lookahead]
x_tmp, prices_test, y_tmp, prices_holdout = train_test_split(X, prices, test_size=0.2, shuffle=False, random_state=1)
x_tmp, x_holdout, y_tmp, y_holdout = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=1)
x_train, x_test, y_train, y_test = train_test_split(x_tmp, y_tmp, test_size=0.2, random_state=1)

#%%
model = Sequential()
#model.add(Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(self.state_size,1)))
model.add(Dense(units=326, input_dim=len(X.values[0]), activation="relu"))
#model.add(Dense(units=256, activation="relu"))
#model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=64, activation="relu"))
#model.add(Dense(units=64, activation="relu"))
#model.add(Dense(units=16, activation="relu"))
#model.add(Dense(units=8, activation="relu"))
#model.add(Dense(units=4, activation="relu"))
#model.add(Dense(units=2, activation="relu"))
model.add(Dense(1, activation="linear"))
model.compile(loss="mse", optimizer=Adam(lr=0.001, decay=1e-6), metrics=['mse','mae'])
BATCH_SIZE = 256
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=35)
mc = ModelCheckpoint('t{}_bs{}.h5'.format(lookahead, BATCH_SIZE), monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# fit model
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=BATCH_SIZE, epochs=400, verbose=True, callbacks=[es, mc])
 
#%%
preds = model.predict(x_holdout)
#%%
offset = 0
amount = 500

plot_1(preds[offset:offset + amount])
plot_1(y_holdout[offset:offset + amount])
plot_1(prices_holdout[offset:offset + amount])

	
#%%
