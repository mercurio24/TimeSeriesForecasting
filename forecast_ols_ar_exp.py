'''
Project: Generic Time Series Forecasting

Contributors:
Fabio Mercurio
'''

#%%
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% XGBOOST But with feature augmentation based on aggregations on time intervals


def OLSRegression(df_input, 
                  target_column, 
                  time_column, 
                  epochs_to_forecast=1, 
                  epochs_to_test=1):
    """
    This function performs regression using feature augmentation and then training XGB with Crossvalidation.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with forecast
    """
    
    X_train = df_input[:-(epochs_to_forecast+epochs_to_test)].drop([time_column,target_column],1)
    y_train = df_input[:-(epochs_to_forecast+epochs_to_test)][target_column]
    
    model = sm.OLS(y_train, X_train)
    res = model.fit()
    print(res.summary())
    
    X_out = df_input[-(epochs_to_forecast+epochs_to_test):].drop([time_column,target_column],1)
    y_out = res.predict(X_out)
    
    forecast = model.predict(res.params, df_input.drop([time_column,target_column],1))
    forecast[: -(epochs_to_forecast+epochs_to_test)] = np.nan
    df_output = df_input.copy()
    df_output["forecast"] = forecast
    df_output["forecast_up"] = forecast * 1.1
    df_output["forecast_low"] = forecast * 0.9
    
    return df_output


def AutoRegression(df_input, 
                  target_column, 
                  time_column, 
                  epochs_to_forecast=1, 
                  epochs_to_test=1,
                  hyper_params_ar={}):
    """
    This function performs regression using feature augmentation and then training XGB with Crossvalidation.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - hyper_params_ar: Parameters of AR model
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with forecast
    """
    
    # create and evaluate an updated autoregressive model
    
    # load dataset
    input_series = df_input[:-(epochs_to_forecast+epochs_to_test)].set_index(time_column, 1)[target_column]
    # split dataset
    model = ar_select_order(input_series, **hyper_params_ar)
    for hyp_param in ["maxlag","glob","ic"]:
        if hyp_param in hyper_params_ar.keys():
            del hyper_params_ar[hyp_param]
            
    model = AutoReg(input_series, lags=model.ar_lags, **hyper_params_ar)
    res = model.fit()
    print(res.summary())
    
    #start_idx = df_input[:-(epochs_to_forecast+epochs_to_test)][time_column].max()
    start_idx = df_input[-(epochs_to_forecast+epochs_to_test):][time_column].min()
    end_idx = df_input[-(epochs_to_forecast+epochs_to_test):][time_column].max()
    
# =============================================================================
#     ### for statsmodels< 0.12.0
#     #forecast_steps = model.predict(res.params, start=start_idx, end=end_idx, dynamic=True)
#     forecast = df_input[target_column] * np.nan
#     forecast[-(epochs_to_forecast+epochs_to_test):] = forecast_steps
#     df_output = df_input.copy()
#     df_output["forecast"] = forecast
#     df_output["forecast_up"] = forecast * 1.1
#     df_output["forecast_low"] = forecast * 0.9
# =============================================================================

    ### for statsmodels>= 0.12.0
    forecast_steps = res.get_prediction(start=start_idx, end=end_idx)
    forecast_steps_mean = forecast_steps.predicted_mean
    forecast_steps_low = forecast_steps.conf_int()["lower"]
    forecast_steps_up = forecast_steps.conf_int()["upper"]
    forecast = df_input[target_column] * np.nan    
    forecast_low = df_input[target_column] * np.nan
    forecast_up = df_input[target_column] * np.nan
    forecast[-(epochs_to_forecast+epochs_to_test):] = forecast_steps_mean
    forecast_low[-(epochs_to_forecast+epochs_to_test):] = forecast_steps_low
    forecast_up[-(epochs_to_forecast+epochs_to_test):] = forecast_steps_up
    df_output = df_input.copy()
    df_output["forecast"] = forecast
    df_output["forecast_low"] = forecast_low
    df_output["forecast_up"] = forecast_up
    
    return df_output


def ExponentialSmoothingRegression(df_input, 
                  target_column, 
                  time_column, 
                  frequency_data,
                  epochs_to_forecast=1, 
                  epochs_to_test=1,
                  smoothing_params={}):
    """
    This function performs regression using feature augmentation and then training XGB with Crossvalidation.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - hyper_params_exp: Parameters of Exponential smoothing model
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with forecast
    """
        
    # load dataset
    input_series = df_input.set_index(time_column, 1)[target_column]
    
    input_series_smoothed = input_series.copy()
    input_series_smoothed.iloc[-(epochs_to_test+epochs_to_forecast):] = np.nan
    input_series_smoothed = input_series_smoothed.ewm(**smoothing_params).mean()
    forecast_steps = input_series_smoothed[-(epochs_to_test+epochs_to_forecast):].values
    
    forecast = df_input[target_column] * np.nan
    forecast[-(epochs_to_forecast+epochs_to_test):] = forecast_steps
    df_output = df_input.copy()
    df_output["forecast"] = forecast
    df_output["forecast_up"] = forecast * 1.1
    df_output["forecast_low"] = forecast * 0.9
    
    return df_output