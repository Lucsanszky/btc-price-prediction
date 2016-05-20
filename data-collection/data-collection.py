
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


# In[2]:

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


# # Transform collected trade data into proper format and export it

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


# # Transform collected order book data into proper format

# In[15]:

path = '../btc-data/BTC_OB_raw.csv'
date_parse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

data = pd.read_csv(path, sep='",')

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
data['Asks'] = data['Asks'].map(lambda x: list(map(lambda t: (float(t[0]), float(t[1])), x)))
data['Bids'] = data['Bids'].map(lambda x: list(map(lambda t: (float(t[0]), float(t[1])), x)))
data.sort_values(by = 'ID', inplace = True)

#data.to_csv(path_or_buf='../btc-data/BTC_OB_clean_test.csv')


# # Recreate the order book - NOT WORKING

# In[129]:

askside = [ (373.1,0.53671),
            (373.23,17.24),
            (373.27,7.9555),
            (373.28,1.3363),
            (373.29,0.11771),
            (373.5,55.04631),
            (373.55,10.6796),
            (373.56,1.43616041),
            (373.59,0.26566839),
            (373.6,1.43882788),
            (373.61,1.43841224),
            (373.65,0.2630484),
            (373.71,1.69908455),
            (373.73,0.43790512),
            (373.75,1.54945824),
            (373.77,1.85661036),
            (373.78,0.17786362),
            (373.8,2.76357342),
            (373.81,1.73747649),
            (373.82,1.16956076)
]
bidside = [(372.85,2.61805646),
           (372.84,8.9237),
           (372.67,8.0285),
           (372.66,3.28509036),
           (372.6,0.02326771),
           (372.25,10.6798),
           (372.24,6.493),
           (372.2,2.0),
           (371.96,6.692),
           (371.74,0.04),
           (371.73,7.265),
           (371.7,0.02231636),
           (371.49,6.928),
           (371.4,0.02233438),
           (371.28,7.441),
           (371.1,0.02235244),
           (371.04,6.84),
           (370.9,0.262),
           (370.88,2.16480323),
           (370.86,45.75)
]


#print(data.tail())
#data.sort_values(by = 'ID', inplace = True)
#print(data.ix[5, 'Asks'])
    
for i in range(int(len(data)/100)):
    #print('Asks: ', askside)
    #print('To update: ', asks)
    
    for ask in data.ix[i, 'Asks']:
        found = False
        for j in range(len(askside)):
            price = askside[j][0]
            vol = askside[j][1]
        if not(found):
            if price == ask[0]:
                if ask[1] == 0:
                    del askside[j]
                else:
                    del askside[j]
                    askside.insert(j, ask)
                found = True
            elif price < ask[0]:
                if ask[1] > 0:
                    askside.insert(j, ask)
                found = True

    #while len(askside) > 20:
        #askside.remove(askside[-1])
            
    for bid in data.ix[i, 'Bids']:
        found = False
        for j in range(len(bidside)):
            price = bidside[j][0]
            vol = bidside[j][1]
        if not(found):
            if price == bid[0]:
                if bid[1] == 0:
                    del bidside[j]
                else:
                    del bidside[j]
                    bidside.insert(j, bid)
                found = True
            elif price > bid[0]:
                if bid[1] > 0:
                    bidside.insert(j, bid)
                found = True

    #while len(bidside) > 20:
        #bidside.remove(bidside[-1])


# In[80]:

askside = {373.1: 0.53671, 
           373.23: 17.24,
           373.27: 7.9555,
           373.28: 1.3363,
           373.29: 0.11771, 
           373.5: 55.04631,
           373.55: 10.6796,
           373.56: 1.43616041, 
           373.59: 0.26566839, 
           373.6: 1.43882788,
           373.61: 1.43841224,
           373.65: 0.2630484, 
           373.71: 1.69908455, 
           373.73: 0.43790512, 
           373.27: 7.9555, 
           373.55: 10.6796, 
           373.22: 1.3363, 373.49: 7.9934, 373.72: 10.6566, 373.28: 1.3363}

asks = {373.27: 7.9929, 373.28: 0.0, 373.49: 0.0, 373.55: 10.6849, 373.72: 0.0}

for ask in asks:
    #print(asks.get(ask))
    if ask in askside:
        if asks.get(ask) == 0:
            askside.pop(ask)
        else:
            askside.pop(ask)
            askside.update({ask: asks.get(ask)})
    elif asks.get(ask) > 0:
        askside.update({ask: asks.get(ask)})
        if len(askside) > 20:
            askside.pop(max(askside))

            
pair = (1,2)
bidside = [(372.85,2.61805646),
           (372.84,8.9237),
           (372.67,8.0285),
           (372.66,3.28509036),
           (372.6,0.02326771),
           (372.25,10.6798),
           (372.24,6.493),
           (372.2,2.0),
           (371.96,6.692),
           (371.74,0.04),
           (371.73,7.265),
           (371.7,0.02231636),
           (371.49,6.928),
           (371.4,0.02233438),
           (371.28,7.441),
           (371.1,0.02235244),
           (371.04,6.84),
           (370.9,0.262),
           (370.88,2.16480323),
           (370.86,45.75)
]
bidside = dict(bidside)
print(bidside)


# # Collect and transform order book states from CryptoIQ

# In[4]:

URL = 'https://cryptoiq.io/api/marketdata/orderbooktop/bitstamp/btcusd/2016-%s'

dates = pd.date_range(start = '1/1/2016', end = '5/1/2016', freq='H')

lob_data = pd.DataFrame()

for date in dates:
    time = str(date.month) + '-' + str(date.day) + '/' + str(date.hour)
    data = pd.read_json(URL % time)
    lob_data = lob_data.append(data)

#lob_data.to_csv(path_or_buf='../btc-data/BTC_LOB_collected.csv')


# # Convert indices to datetime format (rows: 939612)

# In[5]:

lob_data.set_index('time', inplace=True)
lob_data.index = pd.to_datetime(lob_data.index)


# # Create indices for evenly spaced time series (10s)

# In[16]:

dates = pd.date_range(start = '1/1/2016 00:00:00', end = '5/1/2016 00:59:50', freq='10s')
#lob_data.to_csv(path_or_buf='../btc-data/BTC_LOB_collected.csv')


# In[7]:

lob_data['asks'] = lob_data['asks'].map(dict)
lob_data['bids'] = lob_data['bids'].map(dict)


# # Re-index the LOB table with the evenly spaced time series, fill missing values with the nearest available prices (rows: 1045800) 

# In[19]:

lob_data = lob_data.reindex(dates, method = 'nearest')


# In[20]:

lob_data


# In[21]:

lob_data.to_csv(path_or_buf='../btc-data/BTC_LOB_collected.csv')

