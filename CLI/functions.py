# import libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import yfinance as yf
#%matplotlib inline
from datetime import date
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
from pypfopt.expected_returns import mean_historical_return
from pypfopt import risk_models,objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# get the tickers from nasdaq for mage and large companies
def get_tickers():
  df = pd.read_csv('nasdaq_screener.csv')['Symbol']
  return df.tolist()


# Adjusted close for all tickers
def get_data(tickers):
  
  start_year = date.today().year-6
  start_date = str('{}-01-01'.format(start_year))
  end_date = date.today().strftime("%Y-%m-%d")
  
  df = yf.download(tickers, start_date, end_date) 
  if len(df) == 0:
      return df, start_date, end_date
  df = df['Adj Close']
  # fill nan value 
  if isinstance(df,pd.Series):
      df = pd.DataFrame(df)
      df.columns=[tickers.upper()]
      
  df.fillna(axis=1,method='ffill',inplace=True)
  df.fillna(axis=1,method='bfill',inplace = True)
  return df, start_date, end_date

# preprocessing the data build X and y
def data_prep(data,ticker,time_step):
  ticker = ticker.upper()
  df = data.filter([ticker]).values # filter the data by ticker
  X= []
  y =[]
  for i in range(len(df)-time_step): # pre_processing
    X.append(df[i:time_step + i,0]) # features numbers = time_step
    y.append(df[i+time_step,0]) 

  # covnert to array
  X,y = np.array(X),np.array(y) 

  return X,y

# split the data into train and test 
def split_data(X,y,split_percentage):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-split_percentage,shuffle=False)
  return X_train, X_test, y_train, y_test


# scale the data with MinMaxScaler
def scale_data(X_train, X_test, y_train,y_test,time_step):
  scaler = MinMaxScaler(feature_range=(0,1))
  X_train = scaler.fit_transform(X_train)
  y_train = scaler.fit_transform(y_train.reshape(-1,1))
  X_test = scaler.fit_transform(X_test)
  y_test = scaler.fit_transform(y_test.reshape(-1,1))
  return X_train, X_test, y_train,y_test, scaler

#plot the predictions 
def plot_result(data, ticker,predictions,training_shape,model,time_step):
  ticker = ticker.upper()
  data = data.filter([ticker])
  train = data[:training_shape+time_step+1]
  test = data[training_shape+time_step:]
  test['predictions'] = predictions
  plt.figure(figsize=(15,8))
  plt.plot(train[ticker])
  plt.plot(test[[ticker,'predictions']])
  plt.legend(['trian','Val','val_pred'])
  plt.xlabel('Date')
  plt.ylabel('Adj close Price')
  plt.title(f'{model} {ticker} Stock prediction');
  

# prediction for stock in the future 
def prediction_future(X_test,y_test,model,time_step):
  temp1 = X_test[-1][1:].tolist()
  predictions = y_test[-1].tolist()
  temp1.extend(predictions)
  
  for i in range(30):
    if len(temp1)==time_step:
      temp1 = np.array(temp1).reshape(-1,time_step)
      predic = model.predict(temp1)
      temp1 = temp1.tolist()
      predic = predic[0].tolist()
      temp1[0].extend(predic)
      predictions.extend(predic)
      
    else:
      temp1 = temp1[0][1:]
      temp1 = np.array(temp1).reshape(-1,time_step)
      predic = model.predict(temp1)
      temp1 = temp1.tolist()
      predic = predic[0].tolist()
      temp1[0].extend(predic)
      predictions.extend(predic)
  return predictions

# print the prediction percentages
def show_result(predictions,ticker):
  week = float(predictions[7] / predictions[0] *100-100)
  two_week = float(predictions[14] / predictions[0] *100-100)
  month = float(predictions[-1] / predictions[0] *100-100)
  print(f'regarding to price today {round(float(predictions[0]),2)}$ for {ticker} ')
  print(f'predicted stock value for 7 days  is  : {round(week,2)} %')
  print(f'predicted stock value for 14 days is  : {round(two_week,2)} %')
  print(f'predicted stock value for 30 days is  : {round(month,2)} %')
  
  
#build an optimized portfolio 
def port_opt(df,start_date,end_date):
    intention = int(input("input\n1 for maximun sharpe ratio\n2 for minimun risk\n3 for maximum return\n"))
    total_value = int(input('How much would you invest $'))
    print()
    df = df[start_date:end_date]
    mu = mean_historical_return(df)
    S = risk_models.sample_cov(df)
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1)
    if intention == 1:
        ef.max_sharpe()
    elif intention == 2 :
        ef.min_volatility()
    else:
        ef._max_return()
    cleaned_weights = ef.clean_weights()
    ef.portfolio_performance(verbose=True)
    print()
    latest_prices = get_latest_prices(df)
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=total_value)
    allocation, leftover = da.greedy_portfolio()
    name = pd.read_csv('nasdaq_screener.csv')[['Symbol','Name']]
    name.columns = ['Ticker','Name']
    alloc = pd.DataFrame(allocation.items(),columns=['Ticker','Numbers_of_shares'])
    alloc['Allocation %'] = round(alloc['Numbers_of_shares'] / alloc['Numbers_of_shares'].sum(),4)*100
    alloc = pd.merge(alloc,name,how='left',on='Ticker')
  
    if len(alloc) > 0:
        print('Orderd Allocation\n',alloc.to_string())
        print('Left over : $',int(leftover))

    else:
        print('Low money')
        port_opt(df)

  



