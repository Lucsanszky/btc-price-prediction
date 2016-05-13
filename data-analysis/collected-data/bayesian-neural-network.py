
# coding: utf-8

# # Bayesian Neural Network experimentations based on the codes of Thomas Wiecki - NOT WORKING

# # References 
# 
# * http://www.csri.utoronto.ca/~radford/ftp/thesis.pdf
# * https://www.kaggle.com/c/DarkWorlds/details/winners
# * http://blog.kaggle.com/2012/12/19/a-bayesian-approach-to-observing-dark-worlds/
# * http://timsalimans.com/observing-dark-worlds/
# * https://github.com/tqchen/ML-SGHMC/tree/master/bayesnn
# * https://gist.github.com/anonymous/d7d6ee33e06ba1845dda94b5137dfba3
# * http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/tree/master/
# * http://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print11.pdf
# * http://dl.acm.org/citation.cfm?id=1162264
# * http://www.inference.phy.cam.ac.uk/itprnn/book.pdf

# In[1]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information matplotlib, numpy, pandas, pymc3, seaborn, sklearn, theano')


# In[2]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import random
import seaborn as sns
from sklearn import preprocessing as preproc
from sklearn.metrics import mean_squared_error as mse, accuracy_score as acc_scr, mean_absolute_error as mae
import theano
from theano import tensor as T

pd.set_option('html', False)
np.set_printoptions(threshold=np.nan)
sns.set()


# In[3]:

path = '../../btc-data/BTC_LOB_features_10s.csv'
data = pd.read_csv(path, index_col = 0, parse_dates = True)


# # Data Preprocessing

# In[4]:

X, y = data, data['mid price'].copy()

for df in X.columns.tolist():
    X[df] = preproc.StandardScaler().fit(X[df].reshape(-1,1)).transform(X[df].reshape(-1,1))
    
train_dates = X.index[:int(0.3*len(X))]
test_dates = X.index[int(0.3*len(X)):]

#X_train[train_dates[0] : train_dates[-361]] = [X[i : i + 360] for i in range(len(X[:-360]))]
#X_test = X[test_dates[0] : test_dates[-361]]
#y_train = y[train_dates[360] : train_dates[-1]]
#y_test = y[test_dates[360] : test_dates[-361]]


# # Create DataFrames for the training set. Input: mid prices from the previous hour, output: mid price change in the next 10 seconds.

# In[5]:

X_train = pd.DataFrame()
y_train = y[train_dates[360] : train_dates[-1]]

for i in range(360):
    colname = 'mid price ' + str(i + 1)
    X_train[colname] = X['mid price'].ix[i : (len(train_dates) + i - 360)].values


# In[6]:

y_train = list(map(lambda t: 1 if t[1] - t[0] >= 0 else -1, zip(y_train.values, y_train.values[1:])))


# # Bayesian Neural Network setup for ADVI minibatches 

# In[7]:

bnn_in = T.matrix()
bnn_in.tag.test_value = X_train[1:5001]
bnn_out = T.vector()
bnn_out.tag.test_value = y_train[:5000]

n_hidden = 5

# Initialize random but sorted starting weights.
init_1 = np.random.randn(360, n_hidden)
init_1 = init_1[:, np.argsort(init_1.sum(axis=0))]
init_2 = np.random.randn(n_hidden, n_hidden)
init_2 = init_2[:, np.argsort(init_2.sum(axis=0))]
init_out = np.random.randn(n_hidden)
init_out = init_out[np.argsort(init_out)]

    
with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(360, n_hidden), 
                             testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, shape=(n_hidden, n_hidden), 
                             testval=init_2)
    
    # Weights from hidden layer to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, shape=(n_hidden,), 
                              testval=init_out)

    # Build neural-network
    a1 = T.dot(bnn_in, weights_in_1)
    act_1 = T.tanh(a1)
    a2 = T.dot(act_1, weights_1_2)
    act_2 = T.tanh(a2)
    act_out = T.nnet.sigmoid(T.dot(act_2, weights_2_out))
    
    out = pm.Bernoulli('out', 
                        act_out,
                        observed=bnn_out)


# # Bayesian Neural Network setup for MCMC samplings

# In[7]:

bnn_in = T.matrix()
bnn_in.tag.test_value = X_train[1:10001]
bnn_out = T.vector()
bnn_out.tag.test_value = y_train[:10000]

n_hidden = 5

# Initialize random but sorted starting weights.
init_1 = np.random.randn(360, n_hidden)
init_2 = np.random.randn(n_hidden, n_hidden)
init_out = np.random.randn(n_hidden)

with pm.Model() as neural_network:
    # Weights from input to hidden layer
    weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(360, n_hidden), 
                             testval=init_1)
    
    # Weights from 1st to 2nd layer
    weights_1_2 = pm.Normal('w_1_2', 0, sd=1, shape=(n_hidden, n_hidden), 
                             testval=init_2)
    
    # Weights from hidden layer to output
    weights_2_out = pm.Normal('w_2_out', 0, sd=1, shape=(n_hidden,), 
                              testval=init_out)

    # Build neural-network
    act_1 = T.tanh(T.dot(bnn_in, weights_in_1))
    act_2 = T.tanh(T.dot(act_1, weights_1_2))
    act_out = T.nnet.sigmoid(T.dot(act_2, weights_2_out))
    
    out = pm.Bernoulli('out', 
                        act_out,
                        observed=bnn_out)


# # Standard ADVI

# In[8]:

with neural_network:
    # Run ADVI
    means, sds, elbos = pm.variational.advi(n=20000, accurate_elbo=True)


# # Setting up the minibatches

# In[9]:

minibatch_tensors = [bnn_in, bnn_out]
minibatch_RVs = [out]

def create_minibatch(data):
    rng = np.random.RandomState(0)
    
    while True:
        ixs = rng.randint(len(data), size=100)
        yield data[ixs]

minibatches = [
    create_minibatch(X_train[1:10001]), 
    create_minibatch(y_train[:10000]),
]

total_size = len(y_train[:10000])


# # ADVI with minibatches

# In[11]:

with neural_network:
    # Run advi_minibatch
    means, sds, elbos = pm.variational.advi_minibatch(
        n=20000, minibatch_tensors=minibatch_tensors, 
        minibatch_RVs=minibatch_RVs, minibatches=minibatches, 
        total_size=total_size, learning_rate=1e-2, epsilon=1.0
    )


# In[ ]:



