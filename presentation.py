
# coding: utf-8
<body onload="document.getElementById('theme').setAttribute('href','reveal.js/css/theme/serif.css'); return false;">
<style type="text/css">
.reveal h2, .reveal h3, .reveal h4 {
    font-weight: normal;
}
.reveal table, .reveal td, .reveal tr, .reveal th {
    border: 0px solid black;
}
</style>
# In[7]:

from IPython.display import Image
from IPython.display import display


# <center><h1>Ƀ</h1></center>

# #WHAT?

# #WHY?

# #HOW?

# #MACHINE LEARNING TECHNIQUES FOR BITCOIN PRICE PREDICTION

# #EVALUATION METRICS
# 
# * <h3>Coefficient of Determination</h3>
# 
# * <h3>Root-Mean-Square Error</h3>
# 
# * <h3>Mean Absolute Error</h3>
# 
# * <h3>Directional Symmetry</h3>

# #DAILY PRICE PREDICTION - APPROACH
# 
# * <h3><b>DATA:</b> blockchain.info, 4 January 2012 - 13 April 2016 (split: 70/15/15)</h3>
# 
# * <h3>Use a Genetic Algorithm (GA) to select the most relevant Bitcoin network features</h3>
# 
# * <h3>Manually aid the GA feature selection by observing regression and residual plots</h3>
# 
# * <h3>Perform a Multiple Linear Regression with the selected features to predict the prices</h3>

# #FEATURE SELECTION WITH NSGA-II  
# 
# * <h3>Multi-objective optimisation problem:</h3>
#  - <h4>Minimise the feature set size to prevent overfitting</h4> 
#  - <h4>Maximise the predictive power (measured by a selected metric)</h4>
# 
# * <h3><b>FITNESS FUNCTION:</b> Multiple Linear Regression</h3>
# 
# * <h3><b>FITNESS SCORE:</b> determined by the given metric score and the feature set size</h3>

# #REGRESSION PLOTS

# In[3]:

a = Image(filename='images/linear-regression_8_1.png')
b = Image(filename='images/linear-regression_8_2.png')
c = Image(filename='images/linear-regression_8_3.png')
d = Image(filename='images/linear-regression_8_4.png')
e = Image(filename='images/linear-regression_8_5.png')
f = Image(filename='images/linear-regression_8_6.png')
display(a,b,c,d,e,f)


# #MOST COMMONLY SELECTED FEATURES BY THE GA
# 
# * <h4>Mining Revenue</h4>
# * <h4>Network Deficit per Day</h4>
# * <h4>Cost per Transaction</h4>
# * <h4>Total Output Value</h4>
# * <h4>Estimated Transaction Value (USD)</h4>
# * <h4>Trade v Transaction Volume Ratio</h4>
# * <h4>Trade Volume</h4>

# #RESULTS
# 
# | Metric |   Score  |
# |:------:|:--------:|
# |   R2   |  0.6260  |
# |  RMSE  |  47.5497 |
# |   MAE  |  35.5567 |
# | DS     | 53.2189% |

# In[4]:

a = Image(filename='images/linear-regression_13_2.png')
b = Image(filename='images/linear-regression_13_3.png')

display(a,b)


# #RESIDUAL PLOTS
# <img src="images/linear-regression_13_1_copy.png" style="height: 200px;" align = "center"/>
# <img src="images/linear-regression_13_1.png" style="height: 200px;" align = "center"/>

# #INTRADAY PRICE CHANGE PREDICTION - APPROACH
# 
# * <h3><b>DATA:</b></h3>
#  - <h4><b>Limit order book data:</b> cryptoiq.com, 1 January 2016 - 1 May 2016</h4>
#  - <h4><b>Ticker data:</b> bitstamp.net, 1 February 2016 - 24 April 2016</h4>
# 
# * <h3> Time series with frequency of <s>10s</s>, <s>30s</s>, 1m, <b>5m</b>, 10m </h3>
# 
# * <h3>Technical indicators (mainly price-based ones): oscillators, moving averages, indices, etc.</h3>
# 
# * <h3>Window sizes for indicators: 360, 180, 60</h3>
# 
# * <h3>Use Bayesian Ridge Regression to determine the most relevant technical indicators</h3>
# 
# * <h3>Train a Support Vector Regressor with Stochastic Gradient Descent to predict the price changes</h3>

# #BAYESIAN RIDGE REGRESSION 
# 
# * <h3>70/30 split</h3>
# 
# * <h3>All indicators were selected</h3>

# In[9]:

a = Image(filename='images/bayesian-ridge-regression-lob_19_1.png')
b = Image(filename='images/bayesian-ridge-regression-lob_20_1.png')

display(a,b)


# #MID PRICE CHANGE PREDICTION RESULTS
# 
# | Metric |  Score |
# |:------:|:------:|
# |   R2   | 0.0111 |
# |  RMSE  | 0.4493 |
# |   MAE  | 0.2315 |
# | DS     | 56.47% |

# In[8]:

a = Image(filename='images/bayesian-ridge-regression-lob_16_2.png')
b = Image(filename='images/bayesian-ridge-regression-lob_16_3.png')

display(a,b)


# #TRADE PRICE CHANGE PREDICTION RESULTS
# 
# | Metric |  Score |
# |:------:|:------:|
# |   R2   | 0.1184 |
# |  RMSE  | 0.4393 |
# |   MAE  | 0.3103 |
# | DS     | 59.16% |

# In[17]:

a = Image(filename='images/bayesian-ridge-regression-trades_14_2.png')
b = Image(filename='images/bayesian-ridge-regression-trades_14_3.png')

display(a,b)


# #SUPPORT VECTOR REGRESSION WITH STOCHASTIC GRADIENT DESCENT
# 
# * <h3><b>DATA SPLITTING:</b></h3>
#  - <h4>30% for hyperparameter optimisation (20% calibration, 10% validation)</h4>
#  - <h4>40% for offline training</h4>
#  - <h4>30% for testing and online training</h4>
#  
# * <h3>NSGA-II were used for hyperparameter optimisation:</h3>
#  - <h4>Minimise: RMSE</h4>
#  - <h4>Maximise: DS</h4>
#  
# * <h3>Again, all indicators were used</h3>

# #MID PRICE CHANGE PREDICTION RESULTS
# 
# | Metric |  Score  |
# |:------:|:-------:|
# |   R2   | -0.0051 |
# |  RMSE  |  0.4482 |
# |   MAE  |  0.2274 |
# | DS     | 55.53%  |

# In[18]:

a = Image(filename='images/support-vector-regression-lob_17_3.png')
b = Image(filename='images/support-vector-regression-lob_17_4.png')

display(a,b)


# #TRADE PRICE CHANGE PREDICTION RESULTS
# 
# | Metric |  Score |
# |:------:|:------:|
# |   R2   | 0.1064 |
# |  RMSE  | 0.4414 |
# |   MAE  | 0.3042 |
# | DS     | 60.14% |

# In[15]:

a = Image(filename='images/support-vector-regression-trades_15_3.png')
b = Image(filename='images/support-vector-regression-trades_15_4.png')

display(a,b)


# #BACKTESTING - STRATEGY
# 
# * <h3>Holding positions: +1 BTC, 0 BTC, -1 BTC</h3>
# 
# * <h3>Buy 1 BTC when we predict price increase with a previously decreasing price (buy low)</h3>
# 
# * <h3>Sell 1 BTC when we predict price decrease with a previously increasing price (sell high)</h3>
# 
# * <h3>Otherwise: do nothing</h3>

# #BACKTESTING - RESULTS

# In[19]:

a = Image(filename='images/support-vector-regression-trades_18_2.png')
b = Image(filename='images/support-vector-regression-trades_18_3.png')

display(a,b)


# #IMPLEMENTATION
# 
# * <h3>Node.js scripts running on Amazon EC2 instances collecting data and storing it in DynamoDBs</h3>
# 
# * <h3>Online accessible Jupyter notebooks for feature engineering and data analysis, running on Imperial's Cloudstack</h3>
# 
# * <h3>Pandas library for large dataset manipulation</h3>
# 
# * <h3>Scikit-learn for machine learning</h3>
# 
# * <h3>DEAP for evolutionary computation</h3>
# 
# * <h3>Matplotlib and Seaborn for data visualisation </h3>

# #CONCLUSION
# 
# * <h3>Price of BTC mainly depends on trading activity and mining prices</h3>
# 
# * <h3>The Bayesian way of learning seems more successful</h3>
# 
# * <h3>Some additional evaluation is needed</h3>
# 
# * <h3>A scalable model is yet to be developed</h3>

# #FUTURE WORK
# 
# * <h3>Scalabale Bayesian Ridge Regression model</h3>
# 
# * <h3>Bayesian Nerual Networks</h3>
# 
# * <h3>Relevance Vector Machines</h3>

# <center><h1>?</h1></center>

# #EVALUATION METRICS
# 
# * <h4>Coefficient of Determination</h4>
# 
# $$ R^2(y, \hat{y}) = 1 - \frac{\sum_{i = 1}^n (y_i - \hat{y}_i)^2}{\sum_{i = 1}^n(y_i - \bar{y})^2}$$
# 
# * <h4>Root-Mean-Square Error</h4>
# 
# $$ RMSE(y, \hat{y}) = \sqrt{\frac{\sum_{i = 1}^n(y_i - \hat{y}_i)^2}{n}} $$
# 
# * <h4>Mean Absolute Error</h4>
# 
# $$ MAE(y, \hat{y}) = \frac{\sum_{i = 1}^n|y_i - \hat{y}_i|}{n}$$
# 
# * <h4>Directional Symmetry</h4>
# 
# $$DS(y, \hat{y}) = \frac{100}{n} \sum_{i = 1}^n d_i$$
# 
# $$
# d_i =
#   \begin{cases}
#     1, & \quad \text{if } sgn(y_i * \hat{y}_i)\geq 0 \\
#     0, & \quad \text{otherwise}\\
#   \end{cases}
# $$

# #MULTIPLE LINEAR REGRESSION
# 
# Given a vector of inputs $X^T = (x_1, x_2, ..., x_n)$, the model attempts to predict the hypothesis $h(x)$ denoted as follows:
# 
# $$h(x) = \beta_0 + \sum_{i = 1}^n \beta_ix_i$$
# 
# where $\beta_0$ is the bias or intercept and the vector $\beta = (\beta_1, \beta_2, ..., \beta_n)$ is the vector of feature weights learned by the model through a cost function, which we wish to minimise. A commonly used one is the residual sum-of-squares:
# 
# $$RSS(\beta) = \sum_{i = 1}^m(y_i - X_i^T\beta)^2$$
# 
# where m is the number of training data, $y_i$ is the output and $X_i^T\beta$ is the closed form of our hypothesis by setting $x_0$ to 1.

# #BAYESIAN INFERENCE
# 
# $$P(D|S) = \frac{P(D) \times P(S|D)}{P(S)}$$
# 
# Given a set of competing hypotheses which explain a data set, then, for each hypothesis:
# 1. Convert the prior and likelihood information in the data into probabilities
# 2. Multiply them together 
# 3. Normalise the result to get the posterior probability of each hypothesis given the evidence
# 
# Select the most probable hypothesis

# #BAYESIAN RIDGE REGRESSION
# 
# The frequentist formulation of the ridge regression introduces the L2 norm as a penalty to the standard residual sum-of-squares cost function:
# $$PRSS(\beta)_{L2} = \sum_{i = 1}^m(y_i - X_i^T\beta)^2 + \lambda \sum_{j = 1}^n \beta^2$$
# where $\lambda$ is the regularisation parameter. In the Bayesian context, using an L2 penalty is equivalent to setting a Gaussian prior on the weights:
# $$\beta \sim \mathcal{N}(0, \lambda^{-1} \mathbf{I_p})$$

# #SUPPORT VECTOR REGRESSION
# 
# $$ minimise \quad \frac{1}{2}||\mathbf{w}||^2 + C\sum_{i = 1}^n(\xi_i  + \xi_i^*)\\
# s.t. \quad y_i - \mathbf{w}^TX_i - b \leq \epsilon + \xi_i \\
# \quad \mathbf{w}^TX_i + b - y_i \leq \epsilon + \xi_i^* \\ 
# \xi_i, \xi_i^* \geq 0, \quad \forall i = 1,...,n$$

# #STOCHASTIC GRADIENT DESCENT

# In[10]:

Image(filename='images/sgd.png')


# #GENETIC ALGORITHM

# In[11]:

Image(filename='images/ga.png')


# #NETWORK FEATURES

# In[12]:

Image(filename='images/netfeats.png')


# #TECHNICAL INDICATORS

# In[14]:

Image(filename='images/techinds.png')


# https://cloud-vm-46-69.doc.ic.ac.uk:9999

# In[ ]:



