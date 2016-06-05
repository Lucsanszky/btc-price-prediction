
# coding: utf-8

# # Support Vector Regression with Stochastic Gradient Descent (Ticker Data)

# # References
# 
# * http://leon.bottou.org/publications/pdf/compstat-2010.pdf
# * http://research.microsoft.com/pubs/192769/tricks-2012.pdf
# * http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1257413&tag=1
# * http://www.sciencedirect.com/science/article/pii/S0305048301000263
# * http://link.springer.com/book/10.1007/978-3-642-35289-8

# In[4]:

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

get_ipython().magic('load_ext version_information')
get_ipython().magic('version_information deap, matplotlib, numpy, pandas, seaborn, sklearn')


# In[150]:

from deap import base, creator, tools, algorithms
from IPython.display import display
from ipywidgets import widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from sklearn import linear_model
from sklearn import preprocessing as preproc
from sklearn import svm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

np.set_printoptions(threshold=np.nan)
sns.set()
sns.set_color_codes()
sns.set_context("notebook", font_scale=1.35)
toolb = base.Toolbox()


# In[7]:

path = '../btc-data/BTC_Trades_techind_30s.csv'
data30s = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_60s.csv'
data1m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_300s.csv'
data5m = pd.read_csv(path, index_col = 0, parse_dates = True)

path = '../btc-data/BTC_Trades_techind_600s.csv'
data10m = pd.read_csv(path, index_col = 0, parse_dates = True)


# In[8]:

data30s


# In[9]:

data1m


# In[10]:

data5m


# In[11]:

data10m


# In[151]:

def accuracy(act, pred):
    act_ticks = list(map(lambda x: 1 if x >= 0 else 0, act.values))
    pred_ticks = list(map(lambda x: 1 if x >= 0 else 0, pred))
    d = list(map(lambda t: t[0] == t[1], zip(act_ticks, pred_ticks)))
    
    return np.sum(d) / len(act_ticks)


# In[153]:

def fitness_fun(model):
    method, indiv, dataset = model
    
    X_train, X_valid, y_train, y_valid = dataset
    
    # Sometimes the GA assigns the value 0 or less
    # to the parameters, causing the model to fail.
    # The lines below prevent this.
    if indiv[0] <= 0:
        indiv[0] = 0.0001
        
    if indiv[1] <= 0:
        indiv[1] = 0.002
        
    method.alpha = indiv[0]
    method.eta0 = indiv[1]
    
    method.fit(X_train, y_train)
    
    pred = method.predict(X_valid)
    rmse = np.sqrt(mse(y_valid, pred))
    dir_sym = accuracy(y_valid, pred)
    
    return dir_sym, rmse

def nsga2_feat_sel(method, gen_num, indiv_num, dataset):
    # GA configuration
    creator.create("FitnessMulti", base.Fitness, weights = (1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)
    toolb.register('alpha', random.uniform, 1e-8, 0.01)
    toolb.register('eta0', random.uniform, 0.0001, 0.01)
    toolb.register('individual', tools.initCycle, creator.Individual, 
                   (toolb.alpha, toolb.eta0), n = 1)
    toolb.register('population', tools.initRepeat, list, toolb.individual, n = indiv_num)
    toolb.register('evaluate', fitness_fun)
    toolb.register('mate', tools.cxUniform, indpb = 0.1)
    toolb.register('mutate', tools.mutGaussian, mu = 0.0001, sigma = 0.001, indpb = 0.1)
    toolb.register('select', tools.selNSGA2)

    # Initialise the population and the fitness function
    population = toolb.population()
    fits = map (toolb.evaluate, map(lambda x: (method, x, dataset), population))

    # Initialize the placeholder
    # for the best individuals
    hof = tools.HallOfFame(1)
    
    # Run the fitness function on the population
    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    best = np.ndarray((gen_num, 2))

    # Start the evolution
    for gen in range(gen_num):
        offspring = algorithms.varOr(population, toolb, lambda_ = indiv_num, cxpb = 0.55, mutpb = 0.15)
        hof.update(offspring)

        fits = map (toolb.evaluate, map(lambda x: (method, x, dataset), offspring))

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolb.select(offspring + population, k = indiv_num)

        best[gen] = (hof[0].fitness.values)

    chromosome = hof[0]
    
    return best, method, chromosome


# In[154]:

def feature_selection(gen_num, indiv_num, model, dataset):
    results = nsga2_feat_sel(model, gen_num, indiv_num, dataset)
    
    best_model = results[1]
    chromosome = results[2]

    print ('Scores', results[0], '\n')
    print ('Chromosome: ', chromosome, '\n')
    
    # Create dataframes from the metrics
    results_df = pd.DataFrame(results[0], columns = ['Accuracy', 'RMSE'])
    results_df.insert(0, 'Generation', results_df.index)
    
    # Plot the best individuals of each generation based on the metrics
    g = sns.PairGrid(results_df, y_vars=['Accuracy', 'RMSE'], x_vars = 'Generation', size=7, aspect = 2.5)
    g.map(plt.plot)
    
    return best_model, chromosome


# In[155]:

def evaluate(data, features):
    X, y = data[features], data['DELTAP'].copy()
    
    calib_dates = X.index[:int(0.2*len(X))]
    valid_dates = X.index[int(0.2*len(X)):int(0.3*len(X))]
    train_dates = X.index[int(0.3*len(X)):int(0.7*len(X))]
    test_dates = X.index[int(0.7*len(X)):]
    
    X_calib = X[calib_dates[0]:calib_dates[-1]]
    y_calib = y[calib_dates[0]:calib_dates[-1]]
    
    X_valid = X[valid_dates[0]:valid_dates[-1]]
    y_valid = y[valid_dates[0]:valid_dates[-1]]

    X_train = X[train_dates[0]:train_dates[-1]]
    y_train = y[train_dates[0]:train_dates[-1]]
    
    X_test = X[test_dates[0]:test_dates[-1]]
    y_test = y[test_dates[0]:test_dates[-1]]
    
    scaler = preproc.StandardScaler()
    for df in X_calib.columns.tolist():
        scaler.fit(X_calib[df].reshape(-1,1))
        X_calib[df] = scaler.transform(X_calib[df].reshape(-1,1))
        X_valid[df] = scaler.transform(X_valid[df].reshape(-1,1))
        X_train[df] = scaler.transform(X_train[df].reshape(-1,1))
        X_test[df] = scaler.transform(X_test[df].reshape(-1,1))
        
    sgd = linear_model.SGDRegressor(shuffle = True, penalty = 'l2', epsilon = 0,
                                    loss = 'epsilon_insensitive',
                                    n_iter = np.ceil(10**6 / len(X_train)))
    
    dataset = X_calib, X_valid, y_calib, y_valid
    
    model = feature_selection(30, 15, sgd, dataset)
    best_model, chromosome = model
    
    best_model.alpha = chromosome[0]
    best_model.eta0 = chromosome[1]
    best_model.fit(X_train, y_train)
    
    pred = []
    
    # Testing and online learning
    for i in range(len(X_test)):
        x = X_test.ix[i]
        y = y_test.ix[i]
        pred.append(best_model.predict(x.reshape(1,-1)))
        best_model.partial_fit(x.reshape(1,-1), y.ravel(1,))
    
    dr = DummyRegressor(strategy = 'constant', constant = 0)
    dr.fit(X_train, y_train)
    pred_base = dr.predict(X_test)
    rmse_base = np.sqrt(mse(y_test, pred_base))

    print('\n\nResults')
    print('==============================================\n')
    R2_test = best_model.score(X_test, y_test)
    R2_train = best_model.score(X_train, y_train)
    print('Training set R2: ', R2_train, ', Test set R2: ', R2_test)
    rmse_test = np.sqrt(mse(y_test, pred))
    rmse_train = np.sqrt(mse(y_train, best_model.predict(X_train)))
    print('Training set RMSE: ', rmse_train, ', Test set RMSE: ', rmse_test)
    mae_test = mae(y_test, pred)
    mae_train = mae(y_train, best_model.predict(X_train))
    print('Training set MAE: ', mae_train, ', Test set MAE: ', mae_test)
    print('Training set accuracy: ',accuracy(y_train, best_model.predict(X_train)),
          ', Test set accuracy: ', accuracy(y_test, pred), '\n')
    print('Baseline accuracy: ', accuracy(y_test, pred_base))
    print('Baseline RMSE: ', rmse_base)
    print('Mean Price change: ', np.mean(y))
    print('==============================================\n\n')

    plt.figure(figsize = (20,10))
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.ylabel('Price Change')
    plt.legend()
    
    plt.figure(figsize = (20,10))
    plt.plot(y_test.index, y_test, label = 'Actual Price Changes')
    plt.plot(y_test.index, pred, label = 'Predicted Price Changes')
    plt.xlim('2016-04-12 20', '2016-04-13 07')
    plt.ylabel('Price Change')
    plt.legend()


# In[156]:

features = ['K360', 'K180', 'K60', 'D180',
            'D60', 'sD180', 'sD60', 'MOM60',
            'ROC60', 'LWR360', 'LWR180', 'LWR60',
            'ADOSC360', 'ADOSC60', 'DISP360',
            'DISP180', 'DISP60', 'OSCP60-180', 'RSI360',
            'RSI180', 'RSI60', 'CCI180']

evaluate(data30s.copy(), features)


# In[157]:

features = ['K360', 'K180', 'K60', 'D360', 'D180',
            'D60', 'sD180', 'sD60', 'MOM360', 'LWR360',
            'LWR180', 'LWR60', 'ADOSC360', 'ADOSC180',
            'ADOSC60', 'DISP360', 'DISP180', 'DISP60',
            'OSCP180-360', 'OSCP60-180', 'RSI360',
            'RSI180', 'RSI60', 'CCI180'] 

evaluate(data1m.copy(), features)


# In[163]:

features = datas[2].drop(['Price', 'DELTAP'], axis = 1).columns

evaluate(data5m.copy(), features)


# In[167]:

features = data10m.drop(['Price', 'DELTAP'], axis = 1).columns

evaluate(data10m.copy(), features)


# # Backtesting

# In[282]:

features = data5m.drop(['Price', 'DELTAP'], axis = 1).columns

X, y = data5m[features].copy(), data5m['DELTAP'].copy()
prices = data5m['Price'].copy()

train_dates = X.index[:int(0.6*len(X))]
test_dates = X.index[int(0.6*len(X)):]

X_train = X[train_dates[0]:train_dates[-1]]
y_train = y[train_dates[0]:train_dates[-1]]
    
X_test = X[test_dates[0]:test_dates[-1]]
y_test = y[test_dates[0]:test_dates[-1]]

scaler = preproc.StandardScaler()
for df in X_train.columns.tolist():
    scaler.fit(X_train[df].reshape(-1,1))
    X_train[df] = scaler.transform(X_train[df].reshape(-1,1))
    X_test[df] = scaler.transform(X_test[df].reshape(-1,1))
    
sgd = linear_model.SGDRegressor(shuffle = True, penalty = 'l2', epsilon = 0,
                                loss = 'epsilon_insensitive',
                                n_iter = np.ceil(10**6 / len(X_train)),
                                alpha = 0.0004, eta0 = 0.002)

brr = linear_model.BayesianRidge(compute_score=True)

pred_sgd = []
results_sgd = [0]
balance_sgd = 0
coins_sgd = 0

coins_brr = 0
pred_brr = []
results_brr = [0]
balance_brr = 5000


sgd.fit(X_train, y_train)
brr.fit(X_train, y_train)

for i in range(len(X_test) - 1):
    x = X_test.ix[i]
    y = y_test.ix[i]
    prev_sign = np.sign(y_test.ix[i-1])
    change = y_test.ix[i]
    sign_sgd = np.sign(sgd.predict(x.reshape(1,-1)))
    sign_brr = np.sign(brr.predict(x.reshape(1,-1)))
    curr_price = prices[i]
    
    if (sign_sgd == 1 and  prev_sign == -1 and coins_sgd <= 0):
        balance_sgd = y
        coins_sgd = coins_sgd + 1
    elif (sign_sgd == -1 and  prev_sign == 1 and coins_sgd >= 0):
        coins_sgd = coins_sgd - 1
        balance_sgd = y
    else:
        balance_sgd = 0
        
    if (sign_brr == 1 and  prev_sign == -1 and coins_brr <= 0):
        balance_brr = y
        coins_brr = coins_brr + 1
    elif (sign_brr == -1 and  prev_sign == 1 and coins_brr >= 0):
        coins_brr = coins_brr - 1
        balance_brr = y
    else:
        balance_brr = 0
     
    
    results_sgd.append(balance_sgd)
    results_brr.append(balance_brr)
    pred_sgd.append(sgd.predict(x.reshape(1,-1)))
    pred_brr.append(brr.predict(x.reshape(1,-1)))
    sgd.partial_fit(x.reshape(1,-1), y.ravel(1,))
    

fig = plt.figure(figsize = (20,10))
ax1 = fig.add_subplot(111)
ax1.plot(y_test.index, np.cumsum(results_sgd),
         label = 'SVR-SGD Profit (USD)', color = 'r')
ax1.plot(y_test.index, np.cumsum(results_brr),
         label = 'BRR Profit (USD)', color = 'g')
ax1.axhline(color='k')
plt.ylabel('Profit (USD)')
ax2 = ax1.twinx()
ax2.plot(y_test.index,
         prices[test_dates[0]:test_dates[-1]],
         color = 'b', label = 'Price (USD)', alpha = 0.5)
plt.ylabel('Price (USD)')
ax2.set_yticks(np.linspace(ax2.get_yticks()[0],
                           ax2.get_yticks()[-1],
                           len(ax1.get_yticks())))

ax1.legend(loc = 2)
ax2.legend()

fig = plt.figure(figsize = (20, 3))
plt.plot(y_test.index[:-1], y_test[:-1], label = 'Actual Price Changes', alpha = 0.5)
plt.plot(y_test.index[:-1], pred_brr, label = 'Predicted Price Changes (BRR)')
plt.plot(y_test.index[:-1], pred_sgd, label = 'Predicted Price Changes (SVR-SGD)')
plt.ylabel('Price Change')
plt.legend(loc = 4)

accuracy(y_test, pred)


# In[ ]:



