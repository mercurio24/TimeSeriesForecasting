'''
Project: Time Series Forecasting

Contributors:
Fabio Mercurio
'''

#%%
import os
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import arima as pmd_arima
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%% ARIMAX

def pdm_auto_arima(df, 
                   target_column, 
                   time_column, 
                   frequency_data,
                   epochs_to_forecast = 12,
                   d=1,
                   D=0, 
                   seasonal=True,
                   m =12, 
                   start_p = 2, 
                   start_q = 0,
                   max_p=9,
                   max_q=2, 
                   start_P = 0, 
                   start_Q = 0, 
                   max_P = 2, 
                   max_Q = 2,
                   validate = False, 
                   epochs_to_test = 1):
    """
    This function finds the best order parameters for a SARIMAX model, then makes a forecast
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - d, D, m, start_p, start_q, max_p, max_q, start_P, start_Q, max_P, max_Q (int): SARIMAX parameters to be set for reseach 
        - seasonal (bool): seasonality flag
        - validate (bool): if True, epochs_to_test rows are used for validating, else forecast without evaluation
    Returns:
        - forecast_df (pandas.DataFrame): Output DataFrame with best forecast found
    """
    
    assert isinstance(target_column, str)
    assert isinstance(time_column, str)
    
    external_features = [col for col in df if col not in [time_column, target_column]]
    
    if epochs_to_test == 0:
        from warnings import warn
        warn("epochs_to_test=0 and validate=True is not correct, setting validate=False instead")
        validate = False
    
    if frequency_data is not None:
        df = df.set_index(time_column).asfreq(freq=frequency_data, method="bfill").reset_index()
    
    if len(external_features) > 0:
        #Scaling all exogenous features 
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.set_index(time_column).drop([target_column], axis = 1).values)
        
    train_df = df.dropna()
    train_df.set_index(time_column, inplace=True)
    
    if frequency_data is not None:
        date               = pd.date_range(start=df[time_column].min(), periods=len(train_df)+epochs_to_forecast, freq=frequency_data)
    else:
        date               = pd.date_range(start=df[time_column].min(), end=df[time_column].max(), periods=len(df))    
          
    ### Finding parameter using validation set
    if validate:
        train_df_validation = train_df[:-epochs_to_test]
        if len(external_features) > 0:
            exog_validation = scaled[:(len(train_df)-epochs_to_test)]
            model_validation = pmd_arima.auto_arima(train_df_validation[target_column],exogenous = exog_validation, max_order = 30, m=m, 
                              d=d,start_p=start_p, start_q=start_q,max_p=max_p, max_q=max_q, # basic polynomial
                              seasonal=seasonal,D=D, start_P=start_P, max_P = max_P,start_Q = start_Q, max_Q= max_Q, #seasonal polynomial
                              trace=False,error_action='ignore', suppress_warnings=True, stepwise=True)
            exog_validation_forecast = scaled[(len(train_df)-epochs_to_test):len(train_df)]
            forecast_validation, forecast_validation_ci = model_validation.predict(n_periods = epochs_to_test,exogenous= exog_validation_forecast, return_conf_int=True)
            validation_df = pd.DataFrame({target_column:train_df[target_column].values[(len(train_df)-epochs_to_test):len(train_df)],'Forecast':forecast_validation})
            rmse = np.sqrt(mean_squared_error(validation_df[target_column].values, validation_df.Forecast.values))
            print(f'RMSE: {rmse}')
            exog = scaled[:len(train_df)]
            model = pmd_arima.ARIMA(
                              order = list(model_validation.get_params()['order']), 
                              seasonal_order = list(model_validation.get_params()['seasonal_order']),
                              trace=False,error_action='ignore', suppress_warnings=True)
            model.fit(y = train_df[target_column],exogenous = exog)
            training_prediction = model.predict_in_sample(exogenous = exog_validation)
        else: 
            model_validation = pmd_arima.auto_arima(train_df_validation[target_column], max_order = 30, m=m, 
                              d=d,start_p=start_p, start_q=start_q,max_p=max_p, max_q=max_q, # basic polynomial
                              seasonal=seasonal,D=D, start_P=start_P, max_P = max_P,start_Q = start_Q, max_Q= max_Q, #seasonal polynomial
                              trace=False,error_action='ignore', suppress_warnings=True, stepwise=True)
            forecast_validation, forecast_validation_ci = model_validation.predict(n_periods = epochs_to_test, return_conf_int=True)
            validation_df = pd.DataFrame({target_column:train_df[target_column].values[(len(train_df)-epochs_to_test):len(train_df)],'Forecast':forecast_validation})
            rmse = np.sqrt(mean_squared_error(validation_df[target_column].values, validation_df.Forecast.values))
            print(f'RMSE: {rmse}')
            #exog = scaled[:len(train_df)]
            model = pmd_arima.ARIMA(
                              order = list(model_validation.get_params()['order']), 
                              seasonal_order = list(model_validation.get_params()['seasonal_order']),
                              trace=False,error_action='ignore', suppress_warnings=True)
            model.fit(y = train_df[target_column])
            training_prediction = model.predict_in_sample()
    else:
        if len(external_features) > 0:
            #Select exogenous features for training
            exog = scaled[:len(train_df)]
            #Search for best model
            model = pmd_arima.auto_arima(train_df[target_column],exogenous = exog, max_order = 30, m=m, 
                              d=d,start_p=start_p, start_q=start_q,max_p=max_p, max_q=max_q, # basic polynomial
                              seasonal=seasonal,D=D, start_P=start_P, max_P = max_P,start_Q = start_Q, max_Q= max_Q, #seasonal polynomial
                              trace=False,error_action='ignore', suppress_warnings=True, stepwise=True)
            training_prediction = model.predict_in_sample(exogenous = exog) #Training set predictions
        else:
            #Search for best model
            model = pmd_arima.auto_arima(train_df[target_column], max_order = 30, m=m, 
                              d=d,start_p=start_p, start_q=start_q,max_p=max_p, max_q=max_q, # basic polynomial
                              seasonal=seasonal,D=D, start_P=start_P, max_P = max_P,start_Q = start_Q, max_Q= max_Q, #seasonal polynomial
                              trace=False,error_action='ignore', suppress_warnings=True, stepwise=True)
            training_prediction = model.predict_in_sample() #Training set predictions 
    
    ### Forecasting
    if len(external_features) > 0:
        exog_forecast = scaled[len(train_df):len(train_df)+epochs_to_forecast] #Forecast
        if len(exog_forecast)==0:
            exog_forecast = np.nan * np.ones((epochs_to_forecast,exog.shape[1]))    
    if epochs_to_forecast > 0:
        if len(external_features) > 0:
            forecast, forecast_ci = model.predict(n_periods = len(exog_forecast),exogenous= exog_forecast, return_conf_int=True)
        else:
            forecast, forecast_ci = model.predict(n_periods = epochs_to_forecast, return_conf_int=True)

    #Building output dataset
    forecast_df=pd.DataFrame()
    forecast_df[target_column]                                   = df[target_column].values[:len(train_df)+epochs_to_forecast]#df[target_column].values
    forecast_df['forecast']                                 = np.nan
    forecast_df['forecast_up']                              = np.nan
    forecast_df['forecast_low']                             = np.nan
    if validate and epochs_to_forecast > 0:
        forecast_df['forecast'].iloc[-epochs_to_forecast-epochs_to_test:-epochs_to_forecast] = forecast_validation
        forecast_df['forecast_up'].iloc[-epochs_to_forecast-epochs_to_test:-epochs_to_forecast] = forecast_validation_ci[:,1]
        forecast_df['forecast_low'].iloc[-epochs_to_forecast-epochs_to_test:-epochs_to_forecast] = forecast_validation_ci[:,0]
    elif validate and epochs_to_forecast == 0:
        forecast_df['forecast'].iloc[-epochs_to_forecast-epochs_to_test:] = forecast_validation
        forecast_df['forecast_up'].iloc[-epochs_to_forecast-epochs_to_test:] = forecast_validation_ci[:,1]
        forecast_df['forecast_low'].iloc[-epochs_to_forecast-epochs_to_test:] = forecast_validation_ci[:,0]
    if epochs_to_forecast > 0:
        forecast_df['forecast'].iloc[-epochs_to_forecast:]      = forecast
        forecast_df['forecast_up'].iloc[-epochs_to_forecast:]   = forecast_ci[:,1]
        forecast_df['forecast_low'].iloc[-epochs_to_forecast:]  = forecast_ci[:,0]
    forecast_df[time_column]                                     = date
    return forecast_df
    
