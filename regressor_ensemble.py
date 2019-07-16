#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:09:31 2019

@author: lisatostrams
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:05:09 2019

@author: Lisa
"""
import pandas as pd
from sklearn import tree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from tpot import TPOTRegressor
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import RidgeCV
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from catboost import CatBoostRegressor, Pool
from sklearn.feature_selection import SelectPercentile, f_regression

#use certain attributes only for dtc, rf, ridge,knn, svr and general:


#%%
# define the chunks and the features:
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
X.to_csv('ta_features.csv')
#
#scaler = preprocessing.StandardScaler().fit(Xtrain)
#Xtrain_norm = scaler.transform(Xtrain)
#Xval_norm = scaler.transform(Xval)
#X_norm = scaler.transform(X)

prr
classifiers = 'DTC RF REG KNN SVM SVMlinear BRR HR XGB ADA CAT GPR'.split(sep=' ')
#%% train all classifiers
max_depth = 3
n_estimators = 30
tol = 0.001
n_neighbors = 10
        
ytrain_est = np.zeros((len(Xtrain),len(classifiers)))
yval_est = np.zeros((len(Xval),len(classifiers)))
y_est = np.zeros((len(X),len(classifiers)))

dtc = tree.DecisionTreeRegressor(max_depth=max_depth) #train decision tree
dtc = dtc.fit(Xtrain,ytrain)
ytrain_est[:,0] = dtc.predict(Xtrain)
yval_est[:,0] = dtc.predict(Xval)
y_est[:,0] = dtc.predict(X)

rf = RandomForestRegressor(n_estimators = n_estimators,max_depth=max_depth)
rf = rf.fit(Xtrain, ytrain)
ytrain_est[:,1] = rf.predict(Xtrain)
yval_est[:,1] = rf.predict(Xval)
y_est[:,1] = rf.predict(X)

reg = Ridge(alpha = 1)
reg = reg.fit(Xtrain, ytrain)
ytrain_est = reg.predict(Xtrain)
yval_est = reg.predict(Xval)
y_est = reg.predict(X)

tmp = y_est - y.values
plt.plot(tmp[60:])
plt.plot(y_est[-1500:],alpha=0.8)
plt.plot(y.values[-1500:],alpha=0.8)

knn = KNeighborsRegressor(n_neighbors=n_neighbors,algorithm='ball_tree')
knn = knn.fit(Xtrain,ytrain)
ytrain_est[:,3] = knn.predict(Xtrain)
yval_est[:,3] = knn.predict(Xval)
y_est[:,3] = knn.predict(X)


svmnorm = SVR(tol=tol,gamma='auto')
svmnorm = svmnorm.fit(Xtrain_norm, ytrain)
ytrain_est[:,4] = svmnorm.predict(Xtrain_norm)
yval_est[:,4] = svmnorm.predict(Xval_norm)
y_est[:,4] = svmnorm.predict(X_norm)

svmlnorm = LinearSVR(max_iter=10000)
svmlnorm = svmlnorm.fit(Xtrain_norm,ytrain)
ytrain_est[:,5] = svmlnorm.predict(Xtrain_norm)
yval_est[:,5] = svmlnorm.predict(Xval_norm)
y_est[:,5] = svmlnorm.predict(X)

print("processing classifiers, half way")


gnb = BayesianRidge()
gnb = gnb.fit(Xtrain_norm, ytrain)
ytrain_est[:,6] = gnb.predict(Xtrain_norm)
yval_est[:,6] = gnb.predict(Xval_norm)
y_est[:,6] = gnb.predict(X)

hr = HuberRegressor()
hr = hr.fit(Xtrain_norm, ytrain)
ytrain_est[:,7] = hr.predict(Xtrain_norm)
yval_est[:,7] = hr.predict(Xval_norm)
y_est[:,7] = hr.predict(X)

xgb_params = {
        'eta': 0.1,
        'max_depth': max_depth,
        'subsample': 0.75,
        'objective': 'reg:linear',
        'eval_metric': 'mae',
        'silent': True,
        'nthread': 4
}

print("xgb fit")
d_train = xgb.DMatrix(data=Xtrain_norm, label=ytrain, feature_names=Xtrain.columns)
d_val = xgb.DMatrix(data=Xval_norm, label=yval, feature_names=Xval.columns)
d = xgb.DMatrix(data=X,label=y, feature_names=X.columns)
evallist = [(d_val, 'eval'), (d_train, 'train')]
model = xgb.train(dtrain=d_train, num_boost_round=1000, evals=evallist, early_stopping_rounds=50,  params=xgb_params)
ytrain_est[:,8] = model.predict(d_train, ntree_limit=model.best_ntree_limit)
yval_est[:,8] = model.predict(d_val, ntree_limit=model.best_ntree_limit)
y_est[:,8] = model.predict(d,ntree_limit=model.best_ntree_limit)

print("ada fit")
abc = AdaBoostRegressor(n_estimators = n_estimators, learning_rate = 0.05)
abc = abc.fit(Xtrain ,ytrain)
ytrain_est[:,9] = abc.predict(Xtrain)
yval_est[:,9] = abc.predict(Xval)
y_est[:,9] = abc.predict(X)
print("val score: ", np.mean(abs(abc.predict(Xval)-yval)))
print("train_score: ", np.mean((abs(abc.predict(Xtrain)-ytrain))))


print("cat fit")
Cat = CatBoostRegressor(n_estimators=n_estimators, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered")
Cat.fit(Xtrain, 
              ytrain, 
              eval_set=[(Xval, yval)], 
#               eval_metric='mae',
              verbose=2500, 
              early_stopping_rounds=500)

ytrain_est[:,10] = Cat.predict(Xtrain)
yval_est[:,10] = Cat.predict(Xval)
y_est[:,10] = Cat.predict(X)
print("gbm fit")

from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
kernel = DotProduct() + WhiteKernel()

gp = GaussianProcessRegressor(kernel=kernel,alpha=0.001,n_restarts_optimizer=2).fit(Xtrain_norm, ytrain)
ytrain_est[:,11] = gp.predict(Xtrain_norm)
yval_est[:,11] = gp.predict(Xval_norm)
y_est[:,11] = gp.predict(X_norm)



#%%
# Use the TPOT regressor
import datetime    

def test_tpot_performance(test_rounds):  
    scores = []
    current_score = 100
    train_scores = []
    classifiers.append('Tpot')
    old_max = 100
    for i in range(20, 100, 20):  
        print("round: ", i,"/",test_rounds)
        print("start_time = " , datetime.datetime.now().time())
        Tp = TPOTRegressor(max_time_mins =i)
        Tp.fit(Xtrain, ytrain)
        current_score = np.mean(abs(Tp.predict(Xval)-yval))
        train_scores.append(np.mean((abs(Tp.predict(Xtrain)-ytrain))))
        print("score = ", current_score)
        print("current: ", scores)
        if current_score < old_max:
            print("better tpot score")
            Tp.export('tpot_exported_pipeline.py')
            old_max = current_score
        scores.append(current_score)
    plt.subplot(121)
    plt.title('valuation scores')
    plt.plot(scores)
    plt.savefig('Plots/eval_scores.png')
    plt.subplot(122)
    plt.title('Plots/training scores')
    plt.plot(train_scores)
    plt.savefig('train_scores.png')
    plt.show()
    
def test_tpot_MLP(test_rounds):
    scores = []
    current_score = 100
    train_scores = []
    print("classifiers: ", classifiers)    
    for i in range(20, 100, 20):  
        print(i)
        Tp = TPOTRegressor(max_time_mins =i)
        Tp.fit(ytrain_est, ytrain)
        scores.append(np.mean(abs(Tp.predict(yval_est)-yval)))
        train_scores.append(np.mean((abs(Tp.predict(ytrain_est)-ytrain))))
        current_score = np.mean(abs(Tp.predict(yval_est)-yval))
        if current_score == np.min(scores):
            print("score = ", current_score)
            print("scores are: ", scores)
            print("better tpot model")
            Tp.export('tpot_exported_pipeline2.py')
    plt.subplot(121)
    plt.title('valuation scores')
    plt.plot(scores)
    plt.savefig('Plots/eval_scores_mlp.png')
    plt.subplot(122)
    plt.plot(train_scores)
    plt.title('training scores')
    plt.savefig('Plots/train_scores_mlp.png')
    plt.show()
      
    
def use_tpot_as_MLP():
    exported_pipeline = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=3, min_samples_split=20, n_estimators=100)
)
    exported_pipeline.fit(yval_est, yval)
    results = exported_pipeline.predict(predictions)
    return results, exported_pipeline
#%%
# plot feature importance
#mapFeat = dict(zip(["f"+str(i) for i in range(len(features))],features))
#ts = pd.Series(clf.booster().get_fscore())
#ts.index = ts.reset_index()['index'].map(mapFeat)
#ts.order()[-15:].plot(kind="barh", title=("features importance"))

#%%
# classify all loose predictors:
def classify_predictors():
    mseval = np.zeros((len(yval),len(classifiers)))
    msetrain = np.zeros((len(ytrain),len(classifiers)))
    for i in range(len(classifiers)):
        ytrain_est[ytrain_est[:,i]<0,i] = 0
        yval_est[yval_est[:,i]<0,i] = 0
        err = np.mean(abs(ytrain_est[:,i] - ytrain))
        print('Train error for {} is: {:.4f}'.format(classifiers[i],err))
        err = np.mean(abs(yval_est[:,i] - yval))
        print('Test error for {} is: {:.4f}'.format(classifiers[i],err))
        mseval[:,i] = abs(yval_est[:,i] - yval)
        msetrain[:,i] = abs(ytrain_est[:,i] - ytrain)
            
    msevaldf = pd.DataFrame(mseval)
    msetraindf = pd.DataFrame(msetrain)
    print('In total, by selecting the optimal classifier the training MSE is {:.2f}'.format(msetraindf.min(axis=1).mean()))
    print('In total, by selecting the optimal classifier the validation MSE is {:2f}'.format(msevaldf.min(axis=1).mean()))


#%%
# the MLP used to combine the regressors

from sklearn.neural_network import MLPRegressor
def use_mlp():
    models = []
    accuracy = []
    models_sse = []
    sse = []
    score_i = []
    for i in range(1,40):
        model_j = []
        score_j = []
        sse_j = []
        for j in range(0,15):
            clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(i))
            predictie = clf.fit(y_est, y)
            model_j.append(clf)
            score_j.append(np.mean(abs(clf.predict(y_est) - y)))
        print("Layer {} test accuracy: {:.4f}".format(i,min(score_j)))
        
        models.append(model_j[np.argmin(score_j)])
        accuracy.append(min(score_j))
        
    model_acc = np.argmin(accuracy)
    print('Best number of hlayers test acc = {}'.format(model_acc+1)) 
    
    clf = models[model_acc]
    y_hat = clf.predict(y_est)
    y_hat[y_hat<=0] = 0
    return y_hat, clf
#%%

def printStuff():
    print("minimum: ",np.min(y_est))
    print("maximum: ",np.max(y_est))
    print("average: ",np.average(y_est))
    estimatie_train = regressor.predict(ytrain_est)
    verschil_train = np.mean(abs(estimatie_train - ytrain))
    
    estimatie_val = regressor.predict(yval_est)
    print(len(estimatie_val))
    verschil_val = abs((regressor.predict(yval_est) - yval))
    grootste_verschil = np.max(abs((regressor.predict(yval_est) - yval)))
    index_grootste_verschil = np.argmax(grootste_verschil)
    index_grootste_verschil = index_grootste_verschil+1
    
    print("minimum: ",min(estimatie_train))
    print("maximum: ",max(estimatie_train))
    print("average: ",np.average(estimatie_train))
    print("verschil y train = ", np.average(verschil_train))
    print("verschil_val = ", np.average(verschil_val))
    print("grootste_verschil_val = ", grootste_verschil)

    
#%%
def testTpot():
    max = []
    train_max = []
    index = []
    index_j = []
    for i in range(2,10,2):
        for j in range(2,10,2):
            print("i = ", i, " j = ", j)
            Tp = TPOTRegressor(generations=i, population_size=j, cv=5, n_jobs=-1, verbosity =3)
            Tp.fit(Xtrain, ytrain)
            print("val score: ", np.mean(abs(Tp.predict(Xval)-yval)))
            print("train_score: ", np.mean((abs(Tp.predict(Xtrain)-ytrain))))
            max.append(np.mean(abs(Tp.predict(Xval)-yval)))
            index.append(i)
            index_j.append(j)
            train_max.append(np.mean(abs(Tp.predict(Xval)-yval)))
    plt.subplot(121)
    plt.plot(index, max)
    plt.subplot(122)
    plt.plot(index_j, max)
    plt.subplot(212)
    plt.plot(index, train_max)
    plt.subplot(222)
    plt.plot(index_j, train_max)
    plt.show()
    predictions[:,12] = Tp.predict(Xtest)
    ytrain_est[:,12] = Tp.predict(Xtrain)
    yval_est[:,12] = Tp.predict(Xval)
    
        
    
    
#%%
classify_predictors()
y_hat, clf = use_mlp()
#testTpot()
'''mlp = True
test_tpot = True
submit = True
tests=20
if test_tpot:
    #test_tpot_performance(tests)
    test_tpot_MLP(tests)
else:
    print("no testing today")

    
if submit:
    print("using tpot")
    y_est, regressor = use_tpot_as_MLP()
    printStuff()
    print("making submission")
    submission = pd.DataFrame(index=Test.index,columns=['seg_id','time_to_failure'])
    submission['seg_id'] = Test['seg_id'].values
    submission['time_to_failure'] = y_est
    submission.to_csv('submission.csv',index=False)

else:
    print("done")
'''

#%%

plt.plot(y,'b',alpha=0.8)
plt.plot(y_hat,'r',alpha=0.5)
plt.savefig('prediction_bitcoin.png',dpi=300)


#%%
y_hat = pd.Series(y_hat,index=X.index)
y_hat_28 =y_hat.shift(28)
buy = y_hat_28 > X['Close']
sell = y_hat_28 < X['Close']
buy_prices = X['Close'].copy()
buy_prices[~buy.values] =0
sell_prices = X['Close'].copy()
sell_prices[~sell.values] = 0

#%%

plt.plot(X['Close'])
plt.plot(y_hat_28)
plt.plot(buy*1000,alpha=0.5)
plt.plot(1000+sell*1000,alpha=0.5)
plt.plot(buy_prices,'g.',alpha=0.3)
plt.plot(sell_prices,'r.',alpha=0.3)
plt.show()
start= idx[0]
capital = buy_prices.sum()
profit = sell_prices.sum()-capital

print('With a starting capital of {:.2f}$ on {}, following these predictions, you would have made {:.2f}$'.format(capital, start, profit))
