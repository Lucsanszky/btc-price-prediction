
# coding: utf-8

# # Bayesian Ridge Regression

# # References 
# 
# * http://www.machinelearning.org/proceedings/icml2004/papers/354.pdf
# * http://blog.applied.ai/bayesian-inference-with-pymc3-part-2/

# In[2]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information matplotlib, numpy, pandas, pymc3, seaborn, sklearn, theano')


# In[3]:

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


# In[4]:

path = '../btc-data/BTC_LOB_features_10s.csv'
data = pd.read_csv(path, index_col = 0, parse_dates = True)


# # Data Preprocessing

# In[5]:

X, y = data, data['mid price'].copy()

for df in X.columns.tolist():
    X[df] = preproc.StandardScaler().fit(X[df].reshape(-1,1)).transform(X[df].reshape(-1,1))
    
train_dates = X.index[:int(0.7*len(X))]
test_dates = X.index[int(0.7*len(X)):]

#X_train[train_dates[0] : train_dates[-361]] = [X[i : i + 360] for i in range(len(X[:-360]))]
#X_test = X[test_dates[0] : test_dates[-361]]
#y_train = y[train_dates[360] : train_dates[-1]]
#y_test = y[test_dates[360] : test_dates[-361]]


# In[6]:

print('First training date: ', train_dates[0])
print('Last training date: ', train_dates[-1])
print('First testing date: ', test_dates[0])
print('Last testing date: ', test_dates[-1])


# # Create DataFrames for the training set. Input: mid prices from the previous hour, output: mid price change in the next 10 seconds.

# In[6]:

X_train = pd.DataFrame()
y_train = y[train_dates[360] : train_dates[-1]]

for i in range(360):
    colname = 'mid price ' + str(i + 1)
    X_train[colname] = X['mid price'].ix[i : (len(train_dates) + i - 360)].values
    
X_train.set_index(train_dates[360:])


# In[8]:

clf = BayesianRidge(compute_score=True)
clf.fit(X_train, y_train)


# In[9]:

plt.figure(figsize=(12, 10))
plt.title("Weights of the model")
plt.plot(clf.coef_, label="Bayesian Ridge estimate")
plt.xlabel("Features")
plt.ylabel("Weights")


# In[10]:

X_test = pd.DataFrame()
y_test = y[test_dates[360] : test_dates[-1]]

for i in range(360):
    colname = 'mid price ' + str(i + 1)
    X_test[colname] = X['mid price'].ix[(i + len(train_dates)) : (len(train_dates) + len(test_dates) + i - 360)].values

X_test.set_index(test_dates[360:])


# In[11]:

pred = clf.predict(X_test)
pd.DataFrame(pred, index = test_dates[360:])


# In[12]:

plt.figure(figsize = (20,10))
plt.plot(y_test.index, y_test, label = 'Actual Prices')
plt.plot(y_test.index, pred, label = 'Predicted Prices')
plt.legend()


# In[21]:

plt.figure(figsize = (20,10))
plt.plot(y_test.index, y_test, label = 'Actual Prices')
plt.plot(y_test.index, pred, label = 'Predicted Prices')
plt.xlim('2016-04-08 19', '2016-04-08 20')
plt.ylim(416.3, 424.1)
plt.legend()


# In[22]:

R2 = clf.score(X_test, y_test)
R2
rmse_test = np.sqrt(mse(y_test, pred))
rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))

print(rmse_test)
print(rmse_train)


# In[23]:

act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
accuracy = np.sum(act_pred_cmp) / len(act_ticks)
accuracy


# In[26]:

for c in X.drop('mid price', axis = 1).columns.tolist():
    X_train[c] = X[c].ix[359 : len(train_dates) - 1].values


# In[29]:

clf = BayesianRidge(compute_score=True)
clf.fit(X_train, y_train)


# In[38]:

plt.figure(figsize=(12, 10))
plt.title("Weights of the model")
plt.plot(clf.coef_, label="Bayesian Ridge estimate")
plt.xlabel("Features")
plt.xlim(350, 372)
plt.ylabel("Values of the weights")


# In[42]:

X_test.columns.tolist()


# In[31]:

for c in X.drop('mid price', axis = 1).columns.tolist():
    X_test[c] = X[c].ix[len(train_dates) + 359 : len(train_dates) + len(test_dates) - 1].values


# In[32]:

pred = clf.predict(X_test)
pd.DataFrame(pred, index = test_dates[360:])


# In[34]:

plt.figure(figsize = (20,10))
plt.plot(y_test.index, y_test, label = 'Actual Prices')
plt.plot(y_test.index, pred, label = 'Predicted Prices')
plt.xlim('2016-04-08 19', '2016-04-08 20')
plt.ylim(416.3, 424.1)
plt.legend()


# In[35]:

R2 = clf.score(X_test, y_test)
R2
rmse_test = np.sqrt(mse(y_test, pred))
rmse_train = np.sqrt(mse(y_train, clf.predict(X_train)))

print(rmse_test)
print(rmse_train)


# In[36]:

act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
accuracy = np.sum(act_pred_cmp) / len(act_ticks)
accuracy


# In[7]:

X_train = pd.DataFrame()
y_train = y[train_dates[2] : train_dates[-1]]

for i in range(2):
    colname = 'mid price ' + str(i + 1)
    X_train[colname] = X['mid price'].ix[i : (len(train_dates) + i - 2)].values
    
for c in ['ask price', 'bid price', 'mean ask price', 'mean bid price']:
    X_train[c] = X[c].ix[1 : len(train_dates) - 1].values
    
X_train.set_index(train_dates[2:])


# In[8]:

X_test = pd.DataFrame()
y_test = y[test_dates[2] : test_dates[-1]]

for i in range(2):
    colname = 'mid price ' + str(i + 1)
    X_test[colname] = X['mid price'].ix[(i + len(train_dates)) : (len(train_dates) + len(test_dates) + i - 2)].values
    
for c in ['ask price', 'bid price', 'mean ask price', 'mean bid price']:
    X_test[c] = X[c].ix[len(train_dates) + 1 : len(train_dates) + len(test_dates) - 1].values
    
X_test.set_index(test_dates[2:])


# In[19]:

clf = BayesianRidge(compute_score=True)
clf.fit(X_train, y_train)


# In[20]:

pred = clf.predict(X_test)
pd.DataFrame(pred, index = test_dates[2:])


# In[21]:

act_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_test.values, y_test.values[1:])))
pred_ticks = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(pred, pred[1:])))
act_pred_cmp = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
accuracy = np.sum(act_pred_cmp) / len(act_ticks)
accuracy


# In[ ]:



