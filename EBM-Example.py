# Explainable Boosting Machine example code
# by: @CCnossen
# 23 October, 2022

# installation of required packages
#pip install pandas
#pip install interpret
#pip install ta
#pip install yfinance

# import packages
import yfinance as yf
import pandas as pd
import ta as ta
import matplotlib.pyplot as plt
import os

from datetime import datetime, timedelta
from pathlib import Path

# set path for inputdata folder here
absolute_path = '/Users/christian/desktop/EBM/Inputdata/'
os.chdir(absolute_path)

# ------------------------------------------------
# data staging
# ------------------------------------------------

#read in historical BTC prices from csv
btc_hist = pd.read_csv('BTC 2014-2022.csv')

btc_hist['Date'] = pd.to_datetime(btc_hist['Date'])

btc_hist.columns = ['open_time'
              ,'open'
              ,'high'
              ,'low'
              ,'close'
              ,'ignore'
              ,'volume'
              ]


#read in BTC from binance
api_call = pd.read_json (r'https://testnet.binancefuture.com/fapi/v1/klines?symbol=btcusdt&interval=1d&limit=1500')
api_call.columns = ['open_time'
              ,'open'
              ,'high'
              ,'low'
              ,'close'
              ,'volume'
              ,'close_time'
              ,'quote_asset_volume'
              ,'num_trades'
              ,'taker_base_vol'
              ,'taker_quote_vol'
              ,'ignore']

api_call['close_time'] = pd.to_datetime(api_call['close_time'], unit = 'ms')
api_call['open_time'] = pd.to_datetime(api_call['open_time'], unit = 'ms')

# read in sp500 from yfinance
sp500 = yf.download("^GSPC", start = "2014-09-17")
sp500['open_time'] = sp500.index
sp500.columns = ['open'
              ,'high'
              ,'low'
              ,'close_sp500'
              ,'ignore'
              ,'volume'
              ,'open_time'
              ]

sp500 = sp500.drop(columns=['open','high','low','ignore','volume'])

sp500['open_time'] = pd.to_datetime(sp500['open_time'])

# read in Nasdaw composite from yfinance
ndq = yf.download("^IXIC", start = "2014-09-17")
ndq['open_time'] = ndq.index
ndq.columns = ['open'
              ,'high'
              ,'low'
              ,'close_ndq'
              ,'ignore'
              ,'volume'
              ,'open_time'
              ]

ndq = ndq.drop(columns=['open','high','low','ignore','volume'])
ndq['open_time'] = pd.to_datetime(ndq['open_time'])


# read in 10 year treasury's from yfinance
trx = yf.download("^TNX", start = "2014-09-17")
trx['open_time'] = trx.index
trx.columns = ['open'
              ,'high'
              ,'low'
              ,'close_trx'
              ,'ignore'
              ,'volume'
              ,'open_time'
              ]

trx = trx.drop(columns=['open','high','low','ignore','volume'])
trx['open_time'] = pd.to_datetime(trx['open_time'])

print('data staging complete')


# ------------------------------------------------
# Data prep (join)
# ------------------------------------------------

# Join BTC to historical data 
api_call = pd.concat([api_call, btc_hist], axis=0)
api_call = api_call.drop_duplicates(subset=['open_time'])


# join BTC to NDQ and SPX
data = api_call.merge(sp500, on='open_time', how='left')
data = data.merge(ndq, on='open_time', how='left')
data = data.merge(trx, on='open_time', how='left')

# cleanup
data = data.sort_values(by=['open_time'])

del data['close_time']
del data['taker_base_vol']
del data['taker_quote_vol']
del data['ignore']

data['close_sp500'] = data['close_sp500'].fillna(method = 'ffill') #fill the empty days since sp500 is weekday only
data['close_ndq'] = data['close_ndq'].fillna(method = 'ffill') #fill the empty days since sp500 is weekday only
data['close_trx'] = data['close_trx'].fillna(method = 'ffill') #fill the empty days since sp500 is weekday only


print('data prep complete')


# ------------------------------------------------
# feature engineering
# ------------------------------------------------

# helper cols
data['btc_prev'] = data.close.shift(1)
data['sp500_prev'] = data.close_sp500.shift(1)
data['ndq_prev'] = data.close_ndq.shift(1)
data['trx_prev'] = data.close_trx.shift(1)
data['vol_prev'] = data.volume.shift(1)


# f1 = btc change
data['btc_change'] = (data['close'] - data['btc_prev']) / data['btc_prev']
data['f1_btc_change_prev'] = data.btc_change.shift(1)

# f2 = sp500 change
data['sp500_change'] = (data['close_sp500'] - data['sp500_prev']) / data['sp500_prev']
data['f2_sp500_change_prev'] = data.sp500_change.shift(1)
del data['sp500_change']

# f3 = btc RSI 
data['f3_btc_rsi14'] = ta.momentum.RSIIndicator(close = data.btc_prev, window = 14).rsi()

# f4 = sp500 RSI 
data['f4_sp500_rsi14'] = ta.momentum.RSIIndicator(close = data.sp500_prev, window = 14).rsi()

# f5 = btc percentage up/under SMA50
data['btc_sma'] = ta.trend.sma_indicator(close = data.btc_prev, window=50, fillna=False)
data['f5_btc_sma50_perc'] = (data['close'] - data['btc_sma']) / data['btc_sma'] 
del data['btc_sma']

# f6 = btc percentage up/under SMA100
data['btc_sma'] = ta.trend.sma_indicator(close = data.btc_prev, window=100, fillna=False)
data['f6_btc_sma100_perc'] = (data['close'] - data['btc_sma']) / data['btc_sma'] 
del data['btc_sma']

# f7 = force index
data['f7_btc_vol_force_index'] = ta.volume.force_index(close = data.btc_prev, volume = data.vol_prev, window = 13)

# f8 = ndq change
data['ndq_change'] = (data['close_ndq'] - data['ndq_prev']) / data['ndq_prev']
data['f8_ndq_change_prev'] = data.ndq_change.shift(1)
del data['ndq_change']

# f9 = sp500 percentage up/under SMA50
data['sp500_sma'] = ta.trend.sma_indicator(close = data.sp500_prev, window=50, fillna=False)
data['f9_sp500_sma50_perc'] = (data['close_sp500'] - data['sp500_sma']) / data['sp500_sma'] 
del data['sp500_sma']

# f10 = ndq percentage up/under SMA50
data['ndq_sma'] = ta.trend.sma_indicator(close = data.ndq_prev, window=50, fillna=False)
data['f10_ndq_sma50_perc'] = (data['close_ndq'] - data['ndq_sma']) / data['ndq_sma'] 
del data['ndq_sma']

# f11 = btc vs arima 2
data['btc_arima2'] = data.close.shift(2)
data['btc_arima2perc'] = (data['close'] - data['btc_arima2']) / data['btc_arima2']
data['f11_btc_arima2'] = data.btc_arima2perc.shift(1)
del data['btc_arima2']
del data['btc_arima2perc'] 

# f12 = btc vs arima 3
data['btc_arima3'] = data.close.shift(3)
data['btc_arima3perc'] = (data['close'] - data['btc_arima3']) / data['btc_arima3']
data['f12_btc_arima3'] = data.btc_arima3perc.shift(1)
del data['btc_arima3']
del data['btc_arima3perc'] 

# f13 = btc vs arima 4
data['btc_arima4'] = data.close.shift(4)
data['btc_arima4perc'] = (data['close'] - data['btc_arima4']) / data['btc_arima4']
data['f13_btc_arima4'] = data.btc_arima4perc.shift(1)
del data['btc_arima4']
del data['btc_arima4perc'] 

# f14 = btc vs arima 5
data['btc_arima5'] = data.close.shift(5)
data['btc_arima5perc'] = (data['close'] - data['btc_arima5']) / data['btc_arima5']
data['f14_btc_arima5'] = data.btc_arima5perc.shift(1)
del data['btc_arima5']
del data['btc_arima5perc'] 

# f15 = trx change
data['trx_change'] = (data['close_trx'] - data['trx_prev']) / data['trx_prev']
data['f15_trx_change_prev'] = data.trx_change.shift(1)
del data['trx_change']

# f16 = trx percentage up/under SMA50
data['trx_sma'] = ta.trend.sma_indicator(close = data.trx_prev, window=50, fillna=False)
data['f16_trx_sma50_perc'] = (data['close_trx'] - data['trx_sma']) / data['trx_sma'] 
del data['trx_sma']


print('feature engineering complete')


#-------------------------------------------------
# preprocessing
#-------------------------------------------------

# create dependent variable (week-forward close)
data['btc_nextweek'] = data.close.shift(-6)
data['btc_change_nextweek'] = (data['btc_nextweek'] - data['btc_prev']) / data['btc_prev']

# note: prediction variable set here
data['binary'] = data['btc_change_nextweek'].apply(lambda x: 1 if x > 0.1 else '0') 
data['binary'] = data['binary'].astype(float)

del data['btc_nextweek']
del data['btc_change_nextweek']

# remove missing data
data = data.dropna(subset=['f9_sp500_sma50_perc', 'f10_ndq_sma50_perc', 'f6_btc_sma100_perc', 'f16_trx_sma50_perc'])

print('preprocessing complete')

# ------------------------------------------------
# Data optimization
# ------------------------------------------------

# resolve class imbalance problem (16,7% true positive in train/test set)
data_false_1 = data[data['binary']==0]
data_false_2 = data_false_1.sample(frac = 0.33, random_state = 24) #new class imbalance = 25% true positives
data_true_1 = data[data['binary']==1]

data_2 = pd.concat([data_false_2, data_true_1])

data_backup = data
data = data_2
del data_2

# ------------------------------------------------
# Run EBM model
# ------------------------------------------------

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

# split train/test and validate
cutoff_date = max(data.open_time) - timedelta(days = 180) #note: validation set split set here
data_tt = data.loc[(data['open_time'] <= cutoff_date)]
data_vd = data.loc[(data['open_time'] > cutoff_date)]

#reset indexes because otherwise EBM algorithm errors
data_tt = data_tt.reset_index(drop = True)
data_vd = data_vd.reset_index(drop = True)

# run EBM model
train_cols = data_tt.columns[17:33]
label = data_tt.columns[33]
X = data_tt[train_cols]
y = data_tt[label]

# set seed value
seed = 24

# create vectors with train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

ebm = ExplainableBoostingClassifier(random_state=seed
                                    ,max_bins=256 #standard = 256
                                    ,outer_bags=8 #standard = 8
                                    ,inner_bags=0 #standard = 0
                                    )
#fit data
ebm.fit(X_train, y_train)

ebm_global = ebm.explain_global()

# below code will run in jupyter notebook only
#show(ebm_global) 
#ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
#show(ebm_local)

# score values
#ebm.score(X,y)
data_tt['predicted'] = ebm.predict(X) 

# calculate prediction scores for the testset
y_pred = pd.DataFrame(y_test)
y_pred = pd.merge(y_pred, data_tt, left_index = True, right_index = True)
y_pred = y_pred['predicted']

# calculate prediction scores for the trainset
x_pred = pd.DataFrame(y_train)
x_pred = pd.merge(x_pred, data_tt, left_index = True, right_index = True)
x_pred = x_pred['predicted']

# calculate prediction scores for the validationset
Z = data_vd[train_cols]
data_vd['predicted'] = ebm.predict(Z) 
data_vd['predicted_FP'] = ebm.predict_proba(Z).tolist()


# ------------------------------------------------
# Model optimization
# ------------------------------------------------

# insert any good ideas here

# ------------------------------------------------
# Get model metrics
# ------------------------------------------------

from sklearn import metrics
import seaborn as sns
from matplotlib import pyplot

#Confusion Matrix for trainset
cf_train_matrix = metrics.confusion_matrix(y_train, x_pred)
pyplot.figure(figsize = (10,8))
sns.heatmap(cf_train_matrix, annot = True, fmt = 'd')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

a = metrics.confusion_matrix(y_train, x_pred)[1,1] + metrics.confusion_matrix(y_train, x_pred)[0,0]
b = x_pred.count()
model_accuracy = a/b
print('\ntrain accuracy =', model_accuracy)

c = metrics.confusion_matrix(y_train, x_pred)[1,1] 
d = metrics.confusion_matrix(y_train, x_pred)[1,1] + metrics.confusion_matrix(y_train, x_pred)[0,1]
tp_accuracy = c/d
print('train true positives accuracy =', tp_accuracy)


#Confusion Matrix for testset
cf_train_matrix = metrics.confusion_matrix(y_test, y_pred)
pyplot.figure(figsize = (10,8))
sns.heatmap(cf_train_matrix, annot = True, fmt = 'd')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

a = metrics.confusion_matrix(y_test, y_pred)[1,1] + metrics.confusion_matrix(y_test, y_pred)[0,0]
b = y_pred.count()
model_accuracy = a/b
print('\ntest accuracy =', model_accuracy)

c = metrics.confusion_matrix(y_test, y_pred)[1,1]
d = metrics.confusion_matrix(y_test, y_pred)[1,1] + metrics.confusion_matrix(y_test, y_pred)[0,1]
tp_accuracy = c/d
print('test true positives accuracy =', tp_accuracy)


#Confusion Matrix for validation set
cf_train_matrix = metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])
pyplot.figure(figsize = (10,8))
sns.heatmap(cf_train_matrix, annot = True, fmt = 'd')
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')

a = metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])[1,1] + metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])[0,0]
b = data_vd['predicted'].count()
model_accuracy = a/b
print('\nvalidation accuracy =', model_accuracy)

c = metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])[1,1]
d = metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])[1,1] + metrics.confusion_matrix(data_vd['binary'], data_vd['predicted'])[0,1]
tp_accuracy = c/d
print('validation true positives accuracy =', tp_accuracy)
