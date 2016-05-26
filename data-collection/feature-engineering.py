
# coding: utf-8

# # References
# 
# * http://www.sciencedirect.com/science/article/pii/S0925231203003722
# * http://ac.els-cdn.com/S0957417400000270/1-s2.0-S0957417400000270-main.pdf?_tid=3a06fc62-1d5b-11e6-877f-00000aab0f27&acdnat=1463619013_cba9f7ee840313639128ce15571f73ac
# * Technical Analysis of Stock Trends, Robert D. Edwards and John Magee
# * https://www.jbs.cam.ac.uk/fileadmin/user_upload/research/workingpapers/wp0030.pdf
# * http://www.sciencedirect.com/science/article/pii/0261560692900483
# * https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code

# In[3]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information numpy, pandas')


# In[4]:

get_ipython().magic('matplotlib inline')

import ast
import numpy as np
import pandas as pd

np.set_printoptions(threshold=np.nan)


# # Technical Indicator Functions

# In[5]:

def stoch_K(close, window):
    '''Calculates the fast stochastic oscillator %K.
    
    Input:
    close  -- DataFrame to calculate on
    window -- Size of the window
    
    Output: 
    %K -- double
    '''
    
    low = close.rolling(window, center = False).min()
    high = close.rolling(window, center = False).max()
    
    return 100 * (close - low) / (high - low)

def stoch_D(K, window):
    '''Calculates the stochastic oscillator %D.
    %D is the moving average of %K.
    
    Input:
    close  -- DataFrame to calculate on
    window -- Size of the window
    
    Output: 
    %D -- double
    '''
    
    return K.rolling(window, center = False).mean()

def slow_D(D, window):
    '''Calculates the slow stochastic oscillator %D.
    Slow %D is the moving average of %D.
    
    Input:
    close  -- DataFrame to calculate on
    window -- Size of the window
    
    Output: 
    Slow %D -- double
    '''
    
    return D.rolling(window, center = False).mean()

def momentum(close, window):
    '''Calculates the momentum.
    
    Input:
    close  -- DataFrame to calculate on
    window -- Size of the window
    
    Output: 
    Momentum -- double
    '''
    
    dif = lambda x: x[-1] - x[0]
    
    return close.rolling(window, center = False).apply(dif)

def roc(close, window):
    ratio = lambda x: x[-1] / x[0]
    
    return 100 * close.rolling(window, center = False).apply(ratio)

def lw_R(close, window):
    low = close.rolling(window, center = False).min()
    high = close.rolling(window, center = False).max()
    
    return 100 * (high - close) / (high - low)
    
def ad_osc(close, window):
    low = close.rolling(window, center = False).min()
    high = close.rolling(window, center = False).max()
    prev_close = close.rolling(window, center = False).apply(lambda x: x[-2])
    
    return (high - prev_close) / (high - low)

def disp(close, window):
    MA = close.rolling(window, center = False).mean()
    
    return 100 * close / MA

def oscp(close, window1, window2):
    MA1 = close.rolling(window1, center = False).mean()
    MA2 = close.rolling(window2, center = False).mean()
    
    return (MA1 - MA2) / MA1

def rsi(close, window):
    up, down = close.diff().copy(), close.diff().copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    RS = up.rolling(window, center = False).mean() / down.rolling(window, center = False).mean().abs()
    
    return 100 - (100 / (1 + RS))

def cci(close, window):
    low = close.rolling(window, center = False).min()
    high = close.rolling(window, center = False).max()
    
    MT = (close + low + high) / 3
    SMT = MT.rolling(window, center = False).mean()
    DT = MT.rolling(window, center = False).std()
    
    return (MT - SMT) / DT


# # Feature Engineering of LOB Feature Set 1 - simple price and volume features

# In[158]:

lob_data = pd.read_csv('../btc-data/BTC_LOB_collected.csv')


# In[159]:

lob_features10 = pd.DataFrame(lob_data)
lob_features10.set_index(lob_data['Unnamed: 0'], inplace = True)
lob_features10.drop('Unnamed: 0', axis = 1, inplace = True)
lob_features10.index = pd.to_datetime(lob_features10.index)
lob_features10['asks'] = lob_features10['asks'].map(ast.literal_eval)
lob_features10['bids'] = lob_features10['bids'].map(ast.literal_eval)


# In[160]:

lob_features10['total ask volume'] = lob_features10['asks'].map(lambda x: sum(x.values()))
lob_features10['total bid volume'] = lob_features10['bids'].map(lambda x: sum(x.values()))
lob_features10['ask price'] = lob_features10['asks'].map(min)
lob_features10['bid price'] = lob_features10['bids'].map(max)


# In[161]:

lob_features10['bid-ask spread'] = lob_features10['ask price'] - lob_features10['bid price']
lob_features10['mid price'] = (lob_features10['ask price'] + lob_features10['bid price'])/2
lob_features10['ask price spread'] = lob_features10['asks'].map(max) - lob_features10['ask price']
lob_features10['bid price spread'] = lob_features10['bid price'] - lob_features10['bids'].map(min)
lob_features10['mean ask volume'] = lob_features10['total ask volume'] / 20
lob_features10['mean bid volume'] = lob_features10['total bid volume'] / 20
lob_features10['mean ask price'] = lob_features10['asks'].map(sum) / 20
lob_features10['mean bid price'] = lob_features10['bids'].map(sum) / 20


# In[162]:

lob_features10


# In[163]:

lob_features10.drop(['asks', 'bids'], axis = 1, inplace=True)
lob_features10.to_csv(path_or_buf='../btc-data/BTC_LOB_simple_10s.csv')


# In[164]:

lob_features30 = lob_features10.reindex(pd.date_range(start = lob_features10.index[0],
                                                      end = lob_features10.index[-1], freq='30s'))

lob_features60 = lob_features10.reindex(pd.date_range(start = lob_features10.index[0],
                                                      end = lob_features10.index[-1], freq='60s'))

lob_features300 = lob_features10.reindex(pd.date_range(start = lob_features10.index[0],
                                                       end = lob_features10.index[-1], freq='300s'))

lob_features600 = lob_features10.reindex(pd.date_range(start = lob_features10.index[0],
                                                       end = lob_features10.index[-1], freq='600s'))


# In[171]:

lob_features30


# In[172]:

lob_features60


# In[173]:

lob_features300


# In[174]:

lob_features600


# In[175]:

lob_features30.to_csv(path_or_buf='../btc-data/BTC_LOB_simple_30s.csv')
lob_features60.to_csv(path_or_buf='../btc-data/BTC_LOB_simple_60s.csv')
lob_features300.to_csv(path_or_buf='../btc-data/BTC_LOB_simple_300s.csv')
lob_features600.to_csv(path_or_buf='../btc-data/BTC_LOB_simple_600s.csv')


# # Feature Engineering of LOB Feature Set 2 - better technical indicators

# In[ ]:

lob_data = pd.read_csv('../btc-data/BTC_LOB_collected.csv')


# In[274]:

lob_techind10 = pd.DataFrame(lob_features10['mid price'].copy(), index = lob_features10.index)
lob_techind10['B-ASPREAD'] = lob_features10['bid-ask spread'].copy()

lob_techind30 = pd.DataFrame(lob_features30['mid price'].copy(), index = lob_features30.index)
lob_techind30['B-ASPREAD'] = lob_features30['bid-ask spread'].copy()

lob_techind60 = pd.DataFrame(lob_features60['mid price'].copy(), index = lob_features60.index)
lob_techind60['B-ASPREAD'] = lob_features60['bid-ask spread'].copy()

lob_techind300 = pd.DataFrame(lob_features300['mid price'].copy(), index = lob_features300.index)
lob_techind300['B-ASPREAD'] = lob_features300['bid-ask spread'].copy()

lob_techind600 = pd.DataFrame(lob_features600['mid price'].copy(), index = lob_features600.index)
lob_techind600['B-ASPREAD'] = lob_features600['bid-ask spread'].copy()


# In[275]:

def generate_features(frame, freq):
    close = frame['mid price']
    frame['K360'] = stoch_K(close, 360)
    frame['K180'] = stoch_K(close, 180)
    frame['K60'] = stoch_K(close, 60)
    frame['D360'] = stoch_D(frame['K360'], 360)
    frame['D180'] = stoch_D(frame['K180'], 180)
    frame['D60'] = stoch_D(frame['K60'], 60)
    frame['sD360'] = slow_D(frame['D360'], 360)
    frame['sD180'] = slow_D(frame['D180'], 180)
    frame['sD60'] = slow_D(frame['D60'], 60)
    frame['MOM360'] = momentum(close, 360)
    frame['MOM180'] = momentum(close, 180)
    frame['MOM60'] = momentum(close, 60)
    frame['ROC360'] = roc(close, 360)
    frame['ROC180'] = roc(close, 180)
    frame['ROC60'] = roc(close, 60)
    frame['LWR360'] = lw_R(close, 360)
    frame['LWR180'] = lw_R(close, 180)
    frame['LWR60'] = lw_R(close, 60)
    frame['ADOSC360'] = ad_osc(close, 360)
    frame['ADOSC180'] = ad_osc(close, 180)
    frame['ADOSC60'] = ad_osc(close, 60)
    frame['DISP360'] = disp(close, 360)
    frame['DISP180'] = disp(close, 180)
    frame['DISP60'] = disp(close, 60)
    frame['OSCP180-360'] = oscp(close, 180, 360)
    frame['OSCP60-180'] = oscp(close, 60, 180)
    frame['RSI360'] = rsi(close, 360)
    frame['RSI180'] = rsi(close, 180)
    frame['RSI60'] = rsi(close, 60)
    frame['CCI360'] = cci(close, 360)
    frame['CCI180'] = cci(close, 180)
    frame['CCI60'] = cci(close, 60)
    frame['DELTAP'] = close.diff()
    
    frame['mid price'] = frame['mid price'].shift(-1)
    frame['B-ASPREAD'] = frame['B-ASPREAD'].shift(-1)
    frame['DELTAP'] = frame['DELTAP'].shift(-1)
    frame.set_index(frame.index.shift(1, freq=freq), inplace = True)
    frame = frame[3*359:-1]
    
    return frame


# In[276]:

lob_techind10 = generate_features(lob_techind10, '10s')
lob_techind10.replace([np.inf, -np.inf], np.nan, inplace = True)
lob_techind10.fillna(method='ffill', inplace = True)


# In[277]:

lob_techind30 = generate_features(lob_techind30, '30s')
lob_techind30.replace([np.inf, -np.inf], np.nan, inplace = True)
lob_techind30.fillna(method='ffill', inplace = True)


# In[278]:

lob_techind60 = generate_features(lob_techind60, '60s')
lob_techind60.replace([np.inf, -np.inf], np.nan, inplace = True)
lob_techind60.fillna(method='ffill', inplace = True)


# In[279]:

lob_techind300 = generate_features(lob_techind300, '300s')
lob_techind300.replace([np.inf, -np.inf], np.nan, inplace = True)
lob_techind300.fillna(method='ffill', inplace = True)


# In[280]:

lob_techind600 = generate_features(lob_techind600, '600s')
lob_techind600.replace([np.inf, -np.inf], np.nan, inplace = True)
lob_techind600.fillna(method='ffill', inplace = True)


# In[281]:

lob_techind10.to_csv(path_or_buf='../btc-data/BTC_LOB_techind_10s.csv')
lob_techind30.to_csv(path_or_buf='../btc-data/BTC_LOB_techind_30s.csv')
lob_techind60.to_csv(path_or_buf='../btc-data/BTC_LOB_techind_60s.csv')
lob_techind300.to_csv(path_or_buf='../btc-data/BTC_LOB_techind_300s.csv')
lob_techind600.to_csv(path_or_buf='../btc-data/BTC_LOB_techind_600s.csv')


# In[5]:

path = '../btc-data/BTC_LOB_techind_10s.csv'
data10s = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_LOB_techind_30s.csv'
data30s = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_LOB_techind_60s.csv'
data1m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_LOB_techind_300s.csv'
data5m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_LOB_techind_600s.csv'
data10m = pd.read_csv(path, index_col = 0, parse_dates = True)


# In[6]:

data10s


# In[7]:

data30s


# In[8]:

data1m


# In[9]:

data5m


# In[10]:

data10m


# # Feature Engineering of Trades Feature Set

# In[6]:

def generate_trade_features(frame, freq):
    close = frame['Price']
    frame['K360'] = stoch_K(close, 360)
    frame['K180'] = stoch_K(close, 180)
    frame['K60'] = stoch_K(close, 60)
    frame['D360'] = stoch_D(frame['K360'], 360)
    frame['D180'] = stoch_D(frame['K180'], 180)
    frame['D60'] = stoch_D(frame['K60'], 60)
    frame['sD360'] = slow_D(frame['D360'], 360)
    frame['sD180'] = slow_D(frame['D180'], 180)
    frame['sD60'] = slow_D(frame['D60'], 60)
    frame['MOM360'] = momentum(close, 360)
    frame['MOM180'] = momentum(close, 180)
    frame['MOM60'] = momentum(close, 60)
    frame['ROC360'] = roc(close, 360)
    frame['ROC180'] = roc(close, 180)
    frame['ROC60'] = roc(close, 60)
    frame['LWR360'] = lw_R(close, 360)
    frame['LWR180'] = lw_R(close, 180)
    frame['LWR60'] = lw_R(close, 60)
    frame['ADOSC360'] = ad_osc(close, 360)
    frame['ADOSC180'] = ad_osc(close, 180)
    frame['ADOSC60'] = ad_osc(close, 60)
    frame['DISP360'] = disp(close, 360)
    frame['DISP180'] = disp(close, 180)
    frame['DISP60'] = disp(close, 60)
    frame['OSCP180-360'] = oscp(close, 180, 360)
    frame['OSCP60-180'] = oscp(close, 60, 180)
    frame['RSI360'] = rsi(close, 360)
    frame['RSI180'] = rsi(close, 180)
    frame['RSI60'] = rsi(close, 60)
    frame['CCI360'] = cci(close, 360)
    frame['CCI180'] = cci(close, 180)
    frame['CCI60'] = cci(close, 60)
    frame['DELTAP'] = close.diff()
    
    frame['Price'] = frame['Price'].shift(-1)
    frame['DELTAP'] = frame['DELTAP'].shift(-1)
    frame.set_index(frame.index.shift(1, freq=freq), inplace = True)
    frame = frame[3*359:-1]
    
    return frame


# In[7]:

trades = pd.read_csv('../btc-data/BTC_Trades_clean.csv', parse_dates=True)
trades.drop_duplicates(subset = 'Date', inplace = True)
trades.set_index('Date', inplace = True)
trades.drop(['Trade ID', 'Amount'], axis = 1, inplace= True)
trades.index = pd.to_datetime(trades.index)
trades.sort_index(inplace= True)

trades30 = trades.reindex(pd.date_range(start = trades.index[0],
                                        end = trades.index[-1], freq='30s'),
                          method = 'nearest')

trades60 = trades.reindex(pd.date_range(start = trades.index[0],
                                        end = trades.index[-1], freq='60s'),
                          method = 'nearest')

trades300 = trades.reindex(pd.date_range(start = trades.index[0],
                                        end = trades.index[-1], freq='300s'),
                          method = 'nearest')

trades600 = trades.reindex(pd.date_range(start = trades.index[0],
                                        end = trades.index[-1], freq='600s'),
                          method = 'nearest')


# In[8]:

trades_techind30 = generate_trade_features(trades30, '30s')
trades_techind30.replace([np.inf, -np.inf], np.nan, inplace = True)
trades_techind30.fillna(method='ffill', inplace = True)


# In[9]:

trades_techind60 = generate_trade_features(trades60, '60s')
trades_techind60.replace([np.inf, -np.inf], np.nan, inplace = True)
trades_techind60.fillna(method='ffill', inplace = True)


# In[10]:

trades_techind300 = generate_trade_features(trades300, '300s')
trades_techind300.replace([np.inf, -np.inf], np.nan, inplace = True)
trades_techind300.fillna(method='ffill', inplace = True)


# In[11]:

trades_techind600 = generate_trade_features(trades600, '60s')
trades_techind600.replace([np.inf, -np.inf], np.nan, inplace = True)
trades_techind600.fillna(method='ffill', inplace = True)


# In[12]:

trades_techind30.to_csv(path_or_buf='../btc-data/BTC_Trades_techind_30s.csv')
trades_techind60.to_csv(path_or_buf='../btc-data/BTC_Trades_techind_60s.csv')
trades_techind300.to_csv(path_or_buf='../btc-data/BTC_Trades_techind_300s.csv')
trades_techind600.to_csv(path_or_buf='../btc-data/BTC_Trades_techind_600s.csv')


# In[13]:

path = '../btc-data/BTC_Trades_techind_30s.csv'
data30s = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_60s.csv'
data1m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_300s.csv'
data5m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_600s.csv'
data10m = pd.read_csv(path, index_col = 0, parse_dates = True)


# In[14]:

data30s


# In[15]:

data1m


# In[16]:

data5m


# In[17]:

data10m


# In[ ]:



