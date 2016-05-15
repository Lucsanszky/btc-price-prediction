# coding: utf-8
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd
import random
from sklearn import preprocessing as preproc

toolb = base.Toolbox()

# Note: chart names could occasionally change on blockchain.info
URL = 'https://blockchain.info/charts/%s?timespan=all&format=csv'
CHARTS = ['market-price',
          'market-cap',
          'miners-revenue',
          'cost-per-transaction',
          'transaction-fees-usd',
          'network-deficit', 
          'n-transactions', 
          'n-transactions-excluding-popular',
          'n-transactions-excluding-chains-longer-than-10',
          'n-transactions-excluding-chains-longer-than-100',
          'n-transactions-excluding-chains-longer-than-1000',
          'n-transactions-excluding-chains-longer-than-10000',
          'n-unique-addresses', 
          'n-transactions-per-block',
          'n-orphaned-blocks',
          'output-volume',
          'estimated-transaction-volume-usd',
          'trade-volume',
          'tx-trade-ratio',
          'hash-rate',
          'difficulty',
          'median-confirmation-time',
          'bitcoin-days-destroyed',
          'avg-block-size'
         ]

FRAMES = []   # contains everything as DataFrames from charts
FEATURES = [] # standardized DataFrames from charts, excluding market-price

date_parse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y %H:%M:%S')

def prep_data(date_from, date_to):
    del FRAMES[:]
    del FEATURES[:]

    # Create DataFrame from the market-price
    data = pd.read_csv(URL % CHARTS[0], parse_dates=[0], date_parser = date_parse)
    data.columns = ['date', CHARTS[0]]
    
    df = pd.DataFrame(data)
    df['date'] = df['date'].apply(lambda x: x.date())
    df = df.drop_duplicates(['date']).set_index('date').reindex(pd.date_range(start = date_from, end = date_to))
    FRAMES.append(df)

    for chart in CHARTS[1:]:
        data = pd.read_csv(URL % chart, parse_dates=[0], date_parser = date_parse)
        data.columns = ['date', chart]
    
        df = pd.DataFrame(data)
        df['date'] = df['date'].apply(lambda x: x.date())
        df = df.drop_duplicates(['date']).set_index('date').reindex(pd.date_range(start = date_from, end = date_to))
        FRAMES.append(df)

        # Standardize the values inside the DataFrame
        data_np = df.as_matrix()
        scaler = preproc.StandardScaler().fit(data_np)
        data_np_standard = scaler.transform(data_np)

        # Create a new DataFrame from the standardized values
        df_standard = pd.DataFrame(data=data_np_standard, index=df.index, columns=df.columns)
        FEATURES.append(df_standard)

def filter_features(mask):
    return list(map(lambda t: t[1], filter(lambda t: t[0], zip(mask, FEATURES))))

def fitness_fun(model):
    method, metric, indiv = model

    # Sometimes the genetic algorithm produces an all-zero chromosome,
    # which would brake the code. If this 
    if(sum(indiv) == 0):
        indiv[0] = 1
    
    filtered_features = filter_features(indiv)
    size = len(filtered_features)
    filtered_features = pd.concat(filtered_features, axis = 1)
    
    # 70% of the data will be used for training,
    # 15% will be used for validation and testing.
    
    train_dates = filtered_features.index[:int(0.7*len(filtered_features))]

    btc_X_train = filtered_features[train_dates[0] : train_dates[-2]]
    btc_y_train = pd.DataFrame(FRAMES[0])[train_dates[1] : train_dates[-1]]


    valid_dates = filtered_features.index[int(0.7*len(filtered_features)) : int(0.85*len(filtered_features))]
    
    btc_X_valid = filtered_features[valid_dates[0] : valid_dates[-2]]
    btc_y_valid = pd.DataFrame(FRAMES[0])[valid_dates[1] : valid_dates[-1]]

    # Train the learner on the training data
    # and evaluate the performance by the test data

    method.fit(btc_X_train, btc_y_train)
    
    score = metric(btc_X_valid, btc_y_valid)
    
    return score, size

def nsga2_feat_sel(method, metric, objective, gen_num, indiv_num):
    creator.create("FitnessMulti", base.Fitness, weights = objective)
    creator.create("Individual", list, fitness=creator.FitnessMulti) 
    toolb.register('bit', random.randint, 0, 1)
    toolb.register('individual', tools.initRepeat, creator.Individual, toolb.bit, n = len(FEATURES))
    toolb.register('population', tools.initRepeat, list, toolb.individual, n = indiv_num)
    toolb.register('evaluate', fitness_fun)
    toolb.register('mate', tools.cxUniform, indpb = 0.1)
    toolb.register('mutate', tools.mutFlipBit, indpb = 0.05)
    toolb.register('select', tools.selNSGA2)

    population = toolb.population()
    fits = map (toolb.evaluate, map(lambda x: (method, metric, x), population))

    hof = tools.HallOfFame(1)

    for fit, ind in zip(fits, population):
        ind.fitness.values = fit

    best = np.ndarray((gen_num, 1))
    top_RMSE = []

    for gen in range(gen_num):
        offspring = algorithms.varOr(population, toolb, lambda_ = indiv_num, cxpb = 0.5, mutpb = 0.1)
        hof.update(offspring)

        fits = map (toolb.evaluate, map(lambda x: (method, metric, x), offspring))

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolb.select(offspring + population, k = indiv_num)

        best[gen] = hof[0].fitness.values[0]
        top_RMSE = hof[0]

    chromosome = hof[0]
    selected_features = list(map(lambda t: t[1], filter(lambda t: t[0], zip(hof[0], CHARTS[1:]))))
    
    return best, selected_features, chromosome
