
# coding: utf-8

# # References
# 
# * http://www.sciencedirect.com/science/article/pii/S0925231203003722
# * http://ac.els-cdn.com/S0957417400000270/1-s2.0-S0957417400000270-main.pdf?_tid=3a06fc62-1d5b-11e6-877f-00000aab0f27&acdnat=1463619013_cba9f7ee840313639128ce15571f73ac
# * Technical Analysis of Stock Trends, Robert D. Edwards and John Magee
# * https://www.jbs.cam.ac.uk/fileadmin/user_upload/research/workingpapers/wp0030.pdf
# * http://www.sciencedirect.com/science/article/pii/0261560692900483

# In[1]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information deap, matplotlib, numpy, pandas, seaborn, sklearn')


# In[48]:

get_ipython().magic('matplotlib inline')

import ast
from IPython.display import display
from ipywidgets import widgets
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict as od
import random
import re
import seaborn as sns

#pd.set_option('html', False)
np.set_printoptions(threshold=np.nan)
sns.set()


# # Technical Indicators

# In[65]:

def stoch_K(data):
    return 100 * (data[-1] - np.min(data)) / (np.max(data) - np.min(data))

def stoch_D(data):
    K = stoch_K(data)
    
    for i in range(len(data)):
        if i < len(data) - 1:
            K += stoch_K(data)
    
    return K / len(data)

def slow_D(data):
    D = stoch_D(data)
    
    for i in range(len(data)):
        if i < len(data) - 1:
            D += stoch_D(data)
    
    return D / len(data)

def momentum(data):
    return data[-1] - data[0]

def roc(data):
    return 100 * data[-1] / data[0]

def lw_R(data):
    return 100 * (np.max(data) - data[-1]) / (np.max(data) - np.min(data))
    
def ad_osc(data):
    return (np.max(data) - data[-2]) / (np.max(data) - np.min(data))

def disp(data):
    MA = pd.rolling_mean(data, window = len(data), min_periods = 1)
    
    return 100 * data[-1] / MA

def oscp(data1, data2):
    MA1 = pd.rolling_mean(data1, window = len(data1), min_periods = 1)
    MA2 = pd.rolling_mean(data2, window = len(data2), min_periods = 1)
    
    return (MA1 - MA2) / MA1

def rsi(data):
    rev = data.reverse()
    up = list(map(lambda t: t[0] - t[1] if t[0] > t[1] else 0, rev))
    down = list(map(lambda t: t[1] - t[0] if t[0] < t[1] else 0, rev))
    RS = np.mean(up) / np.mean(down)
    
    return 100 - (100 / (1 + RS))


# # Feature Engineering of Feature Set 1 - turned out to be meaningless

# In[3]:

lob_data = pd.read_csv('../btc-data/BTC_LOB_collected.csv')


# In[50]:

lob_features = pd.DataFrame(lob_data)
lob_features.set_index(lob_data['Unnamed: 0'], inplace = True)
lob_features.drop('Unnamed: 0', axis = 1, inplace = True)
lob_features.index = pd.to_datetime(lob_features.index)
lob_features['asks'] = lob_features['asks'].map(ast.literal_eval)
lob_features['bids'] = lob_features['bids'].map(ast.literal_eval)
lob_tech_features = lob_features.copy()


# In[59]:

lob_features['total ask volume'] = lob_features['asks'].map(lambda x: sum(x.values()))
lob_features['total bid volume'] = lob_features['bids'].map(lambda x: sum(x.values()))
lob_features['ask price'] = lob_features['asks'].map(min)
lob_features['bid price'] = lob_features['bids'].map(max)


# In[60]:

lob_features['bid-ask spread'] = lob_features['ask price'] - lob_features['bid price']
lob_features['mid price'] = (lob_features['ask price'] + lob_features['bid price'])/2
lob_features['ask price spread'] = lob_features['asks'].map(max) - lob_features['ask price']
lob_features['bid price spread'] = lob_features['bid price'] - lob_features['bids'].map(min)
lob_features['mean ask volume'] = lob_features['total ask volume'] / 20
lob_features['mean bid volume'] = lob_features['total bid volume'] / 20
lob_features['mean ask price'] = lob_features['asks'].map(sum) / 20
lob_features['mean bid price'] = lob_features['bids'].map(sum) / 20


# In[63]:

lob_features.drop(['asks', 'bids'], axis = 1, inplace=True)


# In[64]:

lob_features.to_csv(path_or_buf='../btc-data/BTC_LOB_features_10s.csv')


# # Feature Engineering of Feature Set 2 - better technical indicators

# In[67]:

lob_tech_features.drop(['asks', 'bids'], axis = 1, inplace = True)
lob_tech_features['mid price'] = lob_features['mid price'].copy()
lob_tech_features


# In[ ]:



