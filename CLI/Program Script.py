# import libraries 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import functions as f 



split_percentage = 0.75
time_step = 60 
all_ticker = f.get_tickers()

while(True):
    for_what= int(input("Enter 1 for Stock predictions\nEnter 2 for Portfolio optimization\nEnter 0 for Rerun\nEnter -1 to Exit\n:"))
    if for_what == -1:
        break
    elif for_what == 1:
        # Stock price prediction
        ticker = input('Enter the ticker:')
        # get the adj close price for all tickers\
        print('Downloading the data')
        print()
        df,start_date, end_date = f.get_data(ticker)
        if len(df) ==0:
            continue
        print()
        print(f'The Data Downloaded From {start_date} to {end_date}')
        print()
        X,y = f.data_prep(df, ticker,time_step)
        X_train, X_test, y_train, y_test = f.split_data(X,y,split_percentage)
        X_train, X_test, y_train,y_test, scaler = f.scale_data(X_train, X_test, y_train, y_test,time_step)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions.reshape(-1,1))
        y_test1 = scaler.inverse_transform(y_test)
        MSE = mean_squared_error(y_test1, predictions)
        #print(f'predicted value on average may be : +/- {round(np.sqrt(MSE),2)}$')
        #f.plot_result(df, ticker, predictions, X_train.shape[0], 'Linear Regression',time_step)
        predictions = f.prediction_future(X_test, y_test, model, time_step)
        
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
        f.show_result(predictions,ticker)
    elif for_what == 2:
        #porfolio Optimization 
        start_date = input('Enter start date in formate yyyy-mm-dd : ')
        end_date = input('Enter end date in formate yyyy-mm-dd : ')
        print('Downloading the data')
        print()
        df,start_date, end_date = f.get_data(all_ticker)
        print()
        print(f'The Data Downloaded From {start_date} to {end_date}')
        print()
        while(True):
            f.port_opt(df,start_date,end_date)
            for_what= int(input("Enter 1 for Stock predictions\nEnter 2 for Portfolio optimization\nEnter 0 for Rerun\nEnter -1 to Exit\n:"))
            if for_what != 2:
                break
            else:
               f.port_opt(df,start_date,end_date) 
    else:
        continue
    
    


















