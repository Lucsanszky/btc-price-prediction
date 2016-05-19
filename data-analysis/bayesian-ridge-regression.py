
# coding: utf-8

# # Bayesian Ridge Regression

# # References 
# 
# * http://www.machinelearning.org/proceedings/icml2004/papers/354.pdf
# * http://blog.applied.ai/bayesian-inference-with-pymc3-part-2/

# In[ ]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information matplotlib, numpy, pandas, pymc3, seaborn, sklearn, theano')


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import random
import seaborn as sns
from sklearn import preprocessing as preproc
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error as mse, accuracy_score as acc_scr, mean_absolute_error as mae
import theano
from theano import tensor as T

#pd.set_option('html', False)
np.set_printoptions(threshold=np.nan)
sns.set()


# In[ ]:

path = '../btc-data/BTC_LOB_features_10s.csv'
data10s = pd.read_csv(path, index_col = 0, parse_dates = True)


# In[ ]:

data30s = data10s.reindex(pd.date_range(start = data10s.index[0], end = data10s.index[-1], freq='30s'))
data1m = data10s.reindex(pd.date_range(start = data10s.index[0], end = data10s.index[-1], freq='60s'))
data5m = data10s.reindex(pd.date_range(start = data10s.index[0], end = data10s.index[-1], freq='300s'))
data10m = data10s.reindex(pd.date_range(start = data10s.index[0], end = data10s.index[-1], freq='600s'))

datas = [data10s, data30s, data1m, data5m, data10m]


# # Data Preprocessing

# In[ ]:

def evaluate(data):
    
    X, y = data, data['mid price'].copy()

    for df in X.columns.tolist():
        X[df] = preproc.StandardScaler().fit(X[df].reshape(-1,1)).transform(X[df].reshape(-1,1))

    train_dates = X.index[:int(0.7*len(X))]
    test_dates = X.index[int(0.7*len(X)):]

    #X_train[train_dates[0] : train_dates[-361]] = [X[i : i + 360] for i in range(len(X[:-360]))]
    #X_test = X[test_dates[0] : test_dates[-361]]
    #y_train = y[train_dates[360] : train_dates[-1]]
    #y_test = y[test_dates[360] : test_dates[-361]]

    print('First training date: ', train_dates[0])
    print('Last training date: ', train_dates[-1])
    print('First testing date: ', test_dates[0])
    print('Last testing date: ', test_dates[-1])

    # Create DataFrames for the training set. Input: mid prices from the previous hour, output: mid price change in the next 10 seconds.

    X_train = pd.DataFrame()
    y_train = y[train_dates[360] : train_dates[-1]]

    for i in range(360):
        colname = 'mid price ' + str(i + 1)
        X_train[colname] = X['mid price'].ix[i : (len(train_dates) + i - 360)].values

    X_train.set_index(train_dates[360:], inplace = True)

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plt.title("Weights of the previous prices")
    plt.plot(clf.coef_)
    plt.xlabel("Prices")
    plt.ylabel("Weights")

    X_test = pd.DataFrame()
    y_test = y[test_dates[360] : test_dates[-1]]

    for i in range(360):
        colname = 'mid price ' + str(i + 1)
        X_test[colname] = X['mid price'].ix[(i + len(train_dates)) : (len(train_dates) + len(test_dates) + i - 360)].values

    X_test.set_index(test_dates[360:], inplace = True)

    pred = clf.predict(X_test)
    pd.DataFrame(pred, index = test_dates[360:])

    plt.figure(figsize = (20,10))
    plt.title('Prediction with previous 360 prices')
    plt.plot(y_test.index, y_test, label = 'Actual Prices')
    plt.plot(y_test.index, pred, label = 'Predicted Prices')
    plt.legend()

    plt.figure(figsize = (20,10))
    plt.title('Zoom in')
    plt.plot(y_test.index, y_test, label = 'Actual Prices')
    plt.plot(y_test.index, pred, label = 'Predicted Prices')
    plt.xlim('2016-04-08 19', '2016-04-08 20:30')
    plt.ylim(416.3, 424.1)
    plt.legend()
    
    print('\n\nResults of prediction with previous 360 prices')
    print('==============================================\n')
    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, clf.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
    pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
    act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
    accuracy = np.sum(act_pred_cmp) / len(act_ticks)
    print('Directional Symmetry: ', accuracy, '\n')
    print('==============================================\n\n')

    for c in X.drop('mid price', axis = 1).columns.tolist():
        for i in range(2):
            colname = c + ' ' + str(i + 1)
            X_train[colname] = X[c].ix[358 + i : len(train_dates) + i - 2].values

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20,10))
    plt.title("Weights of the features and previous prices")
    plt.plot(clf.coef_)
    plt.xlabel("Features")
    plt.xlim(350, 383)
    plt.ylabel("Weights")

    for c in X.drop('mid price', axis = 1).columns.tolist():
        for i in range(2):
            colname = c + ' ' + str(i + 1)
            X_test[colname] = X[c].ix[len(train_dates) + 358 + i : len(train_dates) + len(test_dates) + i - 2].values

    pred = clf.predict(X_test)
    pd.DataFrame(pred, index = test_dates[360:])

    plt.figure(figsize = (20,10))
    plt.title('Prediction with features and prices')
    plt.plot(y_test.index, y_test, label = 'Actual Prices')
    plt.plot(y_test.index, pred, label = 'Predicted Prices')
    plt.xlim('2016-04-08 19', '2016-04-08 20')
    plt.ylim(416.3, 424.1)
    plt.legend()

    print('Results of prediction with previous 360 prices and previous 2 features')
    print('======================================================================\n')
    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, clf.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
    pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
    act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
    accuracy = np.sum(act_pred_cmp) / len(act_ticks)
    print('Directional Symmetry: ', accuracy, '\n')
    print('======================================================================\n\n')

    X_train = pd.DataFrame()
    y_train = y[train_dates[2] : train_dates[-1]]
    
    for i in range(2):
        colname = 'mid price ' + str(i + 1)
        X_train[colname] = X['mid price'].ix[i : (len(train_dates) + i - 2)].values
    
    for c in ['ask price', 'bid price', 'mean ask price', 'mean bid price']:
        X_train[c] = X[c].ix[1 : len(train_dates) - 1].values
    
    X_train.set_index(train_dates[2:], inplace = True)

    X_test = pd.DataFrame()
    y_test = y[test_dates[2] : test_dates[-1]]

    for i in range(2):
        colname = 'mid price ' + str(i + 1)
        X_test[colname] = X['mid price'].ix[(i + len(train_dates)) : (len(train_dates) + len(test_dates) + i - 2)].values
    
    for c in ['ask price', 'bid price', 'mean ask price', 'mean bid price']:
        X_test[c] = X[c].ix[len(train_dates) + 1 : len(train_dates) + len(test_dates) - 1].values
            
    X_test.set_index(test_dates[2:], inplace = True)

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    pd.DataFrame(pred, index = test_dates[2:])

    print('Results of prediction with previous 2 prices and selected features')
    print('=========================================================\n')
    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, clf.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
    pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
    act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
    accuracy = np.sum(act_pred_cmp) / len(act_ticks)
    print('Directional Symmetry: ', accuracy, '\n')
    print('=========================================================\n\n')


# In[ ]:

evaluate(datas[0].copy())


# In[ ]:

evaluate(datas[1].copy())


# In[ ]:

evaluate(datas[2].copy())


# In[ ]:

evaluate(datas[3].copy())


# In[ ]:

evaluate(datas[4].copy())


# In[ ]:



