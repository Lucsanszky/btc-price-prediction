
# coding: utf-8

# In[244]:

get_ipython().magic('matplotlib inline')

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


# In[137]:

path = '../btc-data/BTC_Trades_raw.csv'
date_parse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

data = pd.read_csv(path)

data.TradeData = data.TradeData.map(lambda x: float(re.sub('[^0-9,.]', '', x)))
data.TradeID = data.TradeID.map(lambda x: float(re.sub('[^0-9,.]', '', x)))

data.rename(columns = {data.columns[0]: 'Amount', data.columns[1]: 'Price'}, inplace = True)
data.insert(0, 'Trade ID', data.index)

data['Date'] = data['Date'].map(date_parse)

data.set_index('Date', inplace = True)
data.sort_values(by = 'Trade ID', inplace = True)

data.to_csv(path_or_buf='../btc-data/BTC_Trades_clean.csv')


# In[258]:

def generate_ob_frame():
    path = '../btc-data/BTC_OB_raw.csv'
    date_parse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

    data = pd.read_csv(path, sep='",', nrows=5000)

    data.rename(columns = {data.columns[0]: 'LOB Data', 
                           data.columns[1]: 'ID', 
                           data.columns[2]: 'Date'}, inplace = True)

    data['LOB Data'] = data['LOB Data'].map(lambda x: re.sub('asks', 'asks ', x))
    data['LOB Data'] = data['LOB Data'].map(lambda x: re.sub('bids', 'bids ', x))
    data['LOB Data'] = data['LOB Data'].map(lambda x: re.sub('[^0-9,.,asks ,bids ,]', '', x))
    data['ID'] = data['ID'].map(lambda x: re.sub('\D', '', x))
    data['Date'] = data['Date'].map(lambda x: re.sub('[^0-9,/,:, ]', '', x))

    data['Date'] = data['Date'].map(date_parse)

    bids = data['LOB Data'].map(lambda x: re.split(',', re.sub('bids ', '', re.sub('^asks [0-9,.]*', '', x))))
    asks = data['LOB Data'].map(lambda x: re.split(',', re.sub('asks ', '', re.sub('bids [0-9,.]*', '', x)[:-1])))

    data.insert(1, 'Bids', bids)
    data.insert(1, 'Asks', asks)

    data['Bids'] = data['Bids'].map(lambda x: list(zip(x[::2], x[1::2])))
    data['Asks'] = data['Asks'].map(lambda x: list(zip(x[::2], x[1::2])))

    data.set_index('Date', inplace = True)
    data.drop('LOB Data', axis = 1, inplace = True)

    data = data[['ID','Asks', 'Bids']]
    data['ID'] = data['ID'].map(lambda x: int(x))
    data.sort_values(by = 'ID', inplace = True)

#data.to_csv(path_or_buf='../btc-data/BTC_OB_clean_test.csv')
get_ipython().magic('time generate_ob_frame()')


# In[257]:

askside = []
bidside = []

print(data.tail())

for i in range(len(data)):
    for ask in data.ix[i, 'Asks']:
        if float(ask[0]) in askside:
            if float(ask[1]) == 0:
                askside.remove(float(ask[0]))
            else:
                index = askside.index(ask[0])
                #update volume
                askside[index][1] += ask[1]
        elif len(askside) >= 50:
            if float(ask[1]) != 0:
                if max(askside) > ask[0]:
                    askside.remove(max(askside))
                    askside.append((float(ask[0]), float(ask[1])))
        else:
            if float(ask[1]) != 0:
                askside.append((float(ask[0]), float(ask[1])))
    
    for bid in data.ix[i, 'Bids']:
        if float(bid[0]) in bidside:
            if float(bid[1]) == 0:
                bidside.remove(float(bid[0]))
            else:
                index = bidside.index(bid[0])
                #update volume
                bidside[index][1] += bid[1]
        elif len(bidside) >= 50:
            if float(bid[1]) != 0:
                if min(bidside) > bid[0]:
                    bidside.remove(min(bidside))
                    bidside.append((float(bid[0]), float(bid[1])))
        else:
            if float(bid[1]) != 0:
                bidside.append((float(bid[0]), float(bid[1])))
        
print(list(askside))
print(len(askside))
print(list(bidside))
print(len(bidside))


# In[242]:

data.Bids = data['Bids'].map(float, re.split(',', re.sub('[^0-9,.]', '', b)))
    
data.head()


# In[ ]:



