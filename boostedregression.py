#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:29:36 2019

@author: lisatostrams
"""

minutes = pd.read_csv("in_a_minute_change.csv")
print('There are {} minutes in the file.'.format(len(minutes)))
#minutes.drop('Unnamed: 0',axis=1,inplace=True)
cols = list(minutes.columns.values)
minutes.index = minutes.Timestamp
cols.remove('Timestamp')
cols.remove('open')
cols.remove('close')
cols.remove('high')
cols.remove('low')
cols.remove('Volume')
cols.remove('Price')
cols.remove('Amount')

X = minutes[cols]
y = minutes['Price'].shift(periods=60)
idx = minutes['Timestamp']
del minutes
X=X.replace([np.inf, -np.inf], np.nan)
X=X.fillna(0)
y=y.fillna(0)

Xtrain,Xval,ytrain,yval = model_selection.train_test_split(X,y,test_size=0.4)
#X.to_csv('ta_features.csv')
#
scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain_norm = scaler.transform(Xtrain)
Xval_norm = scaler.transform(Xval)
X_norm = scaler.transform(X)

reg = Ridge(alpha = 1)
reg = reg.fit(Xtrain, ytrain)
ytrain_est = reg.predict(Xtrain)
yval_est = reg.predict(Xval)
y_est = reg.predict(X)


tmp = y_est - y.values
plt.plot(tmp[60:])
plt.plot(y_est[-1500:],alpha=0.8)
plt.plot(y.values[-1500:],alpha=0.8)