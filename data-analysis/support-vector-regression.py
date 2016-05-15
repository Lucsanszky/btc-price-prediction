
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information deap, matplotlib, numpy, pandas, pymc3, seaborn, sklearn, theano')


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

pd.set_option('html', False)
np.set_printoptions(threshold=np.nan)
sns.set()


# In[ ]:



