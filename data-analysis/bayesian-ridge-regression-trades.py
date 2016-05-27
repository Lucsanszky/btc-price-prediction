
# coding: utf-8

# # Bayesian Ridge Regression (Ticker Data)

# # References 
# 
# * http://www.machinelearning.org/proceedings/icml2004/papers/354.pdf
# * http://blog.applied.ai/bayesian-inference-with-pymc3-part-2/

# In[14]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information matplotlib, numpy, pandas, seaborn, sklearn')


# In[15]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go 
import pymc3 as pm
import random
import seaborn as sns
from sklearn import preprocessing as preproc
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error as mse, accuracy_score as acc_scr, mean_absolute_error as mae

np.set_printoptions(threshold=np.nan)
sns.set()
sns.set_context("notebook", font_scale=1.35)


# In[17]:

path = '../btc-data/BTC_Trades_techind_30s.csv'
data30s = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_60s.csv'
data1m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_300s.csv'
data5m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_600s.csv'
data10m = pd.read_csv(path, index_col = 0, parse_dates = True)


# In[18]:

data30s


# In[19]:

data1m


# In[20]:

data5m


# In[21]:

data10m


# In[23]:

def accuracy(act, pred):
    act_ticks = list(map(lambda x: 1 if x >= 0 else 0, act.values))
    pred_ticks = list(map(lambda x: 1 if x >= 0 else 0, pred))
    d = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
    
    return np.sum(d) / len(act_ticks)


# # Data Preprocessing

# In[24]:

def evaluate(data):
    X, y = data, data['DELTAP'].copy()
        
    train_dates = X.index[:int(0.7*len(X))]
    test_dates = X.index[int(0.7*len(X)):]

    print('First training date: ', train_dates[0])
    print('Last training date: ', train_dates[-1])
    print('First testing date: ', test_dates[0])
    print('Last testing date: ', test_dates[-1])

    X_train = X[train_dates[0]:train_dates[-1]].drop(['Price', 'DELTAP'], axis = 1)
    y_train = y[train_dates[0]:train_dates[-1]]
    
    X_test = X[test_dates[0]:test_dates[-1]].drop(['Price', 'DELTAP'], axis = 1)
    y_test = y[test_dates[0]:test_dates[-1]]
    
    scaler = preproc.StandardScaler()
    for df in X_train.columns.tolist():
        scaler.fit(X_train[df].reshape(-1,1))
        X_train[df] = scaler.transform(X_train[df].reshape(-1,1))
        X_test[df] = scaler.transform(X_test[df].reshape(-1,1))

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plt.title("Weights of the Technical Indicators")
    plt.plot(clf.coef_)
    plt.xlabel("Technical Indicators")
    plt.ylabel("Weights")
    
    selected = list(map(lambda t: t[0],
                       (filter(lambda t: np.abs(t[1]) > 0.005, 
                       zip(X_train.columns, clf.coef_)))))

    pred = clf.predict(X_test)
    pd.DataFrame(pred, index = test_dates)

    plt.figure(figsize = (20,10))
    plt.title('Prediction with all the indicators')
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.ylabel('Price Changes')
    plt.legend()

    plt.figure(figsize = (20,10))
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.xlim('2016-04-12 20', '2016-04-13 07')
    plt.ylabel('Price Changes')
    plt.legend()
    
    print('\n\nResults of prediction with all the technical indicators')
    print('===========================================================\n')
    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, clf.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    print('Training set accuracy: ',accuracy(y_train, clf.predict(X_train)),
          ', Test set accuracy: ', accuracy(y_test, pred), '\n')
    print('===========================================================\n\n')
    print(selected, '\n\n')
    
    X_train = X[selected][train_dates[0]:train_dates[-1]]
    y_train = y[train_dates[0]:train_dates[-1]]

    clf = BayesianRidge(compute_score=True)
    clf.fit(X_train, y_train)

    plt.figure(figsize=(20, 10))
    plt.title("Weights of the selected Technical Indicators")
    plt.plot(clf.coef_)
    plt.xlabel("Technical Indicators")
    plt.ylabel("Weights")

    X_test = X[selected][test_dates[0]:test_dates[-1]]
    y_test = y[test_dates[0]:test_dates[-1]]

    pred = clf.predict(X_test)
    pd.DataFrame(pred, index = test_dates)

    plt.figure(figsize = (20,10))
    plt.title('Prediction with selected indicators')
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.ylabel('Price Changes')
    plt.legend()

    plt.figure(figsize = (20,10))
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.xlim('2016-04-12 20', '2016-04-13 07')
    plt.ylabel('Price Changes')
    plt.legend()
    
    print('\n\nResults of prediction with selected technical indicators')
    print('============================================================\n')
    R2_test = clf.score(X_test, y_test)
    R2_train = clf.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, clf.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    print('Training set accuracy: ',accuracy(y_train, clf.predict(X_train)),
          ', Test set accuracy: ', accuracy(y_test, pred), '\n')
    print('============================================================\n\n')


# In[25]:

evaluate(data30s.copy())


# In[26]:

evaluate(data1m.copy())


# In[27]:

evaluate(data5m.copy())


# In[28]:

evaluate(data10m.copy())


# In[ ]:



