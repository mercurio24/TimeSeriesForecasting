'''
Project: Time series forecasting

Contributors:
Fabio Mercurio
'''

#%% LIBRAIRES AND SETTINGS
import os
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from time import time
### Other functions from custom libraries are imported inside this code


#%% GENERAL CALL

def time_series_forecasting(df_input, 
                            target_column,
                            time_column,
                            frequency_data,
                            epochs_to_forecast,
                            epochs_to_test,
                            method_parameters_inputs,
                            show_minimal_columns=False):
    """
    This function returns the forecast of the input time series according to the parameters.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series with NaNs in the target column in the time steps to forecast.
        - target_column (str): name of the target feature to be predicted
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of future records to be predicted
        - epochs_to_test (int): number of last Timestamps where forecast and actual are overlapping, in order to validate
        - method (str): Time Series Forecast model chosen
        - hyper_params (dict): Dictionary with parameters as keys and lists as values for model tuning
        - n_previous_steps (int): useful only for method="XGB", it says how many previous time steps are used as feature for feature augementation
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with Timestamps, target, forecast and confidence intervals 
    """
       
    columns_to_show = [time_column, target_column, "forecast", "forecast_low", "forecast_up"]
    external_features = [col for col in df_input.columns if col not in [time_column, target_column]]    
    assert epochs_to_forecast + epochs_to_test > 0
    
    #Shortening the input dataset to the number of forecast we need
    if len(external_features)>0:
        if frequency_data not in ["M","MS","Y","YS"]:
            last_known_record_date = max(df_input.set_index(time_column).dropna().index)
            last_forecast_time = last_known_record_date + pd.Timedelta(np.timedelta64(epochs_to_forecast,frequency_data))
            df_input_short = df_input[df_input[time_column] <= last_forecast_time]
            if last_forecast_time > max(df_input[time_column]):
                raise Exception("Forecasting is beyond last exogenous row")
        elif frequency_data in ["M","MS"]:
            last_known_record_date = max(df_input.set_index(time_column).dropna().index)
            last_forecast_time = last_known_record_date + pd.Timedelta(np.timedelta64(epochs_to_forecast*31,"D"))
            df_input_short = df_input[df_input[time_column] <= last_forecast_time]
            if (last_forecast_time.month > max(df_input[time_column]).month) or (last_forecast_time.year > max(df_input[time_column]).year):
                raise Exception("Forecasting is beyond last exogenous row")
        elif frequency_data in ["Y","YS"]:
            last_known_record_date = max(df_input.set_index(time_column).dropna().index)
            last_forecast_time = last_known_record_date + pd.Timedelta(np.timedelta64(epochs_to_forecast*366,"D"))
            df_input_short = df_input[df_input[time_column] <= last_forecast_time]
            if last_forecast_time.year > max(df_input[time_column]).year:
                raise Exception("Forecasting is beyond last exogenous row")
    else:
        print("No external features as input. No limit on steps to forecast.")
        #index = pd.date_range('1/1/2000', periods=4, freq='T')
        last_known_record_date = max(df_input.set_index(time_column).dropna().index)
        last_forecast_time = last_known_record_date + pd.Timedelta(np.timedelta64(epochs_to_forecast,frequency_data))
        df_input_short = df_input[df_input[time_column] <= last_forecast_time]
        if last_forecast_time > max(df_input[time_column]):
            extended_index = pd.Series(pd.date_range(min(df_input[time_column]), periods=len(df_input[target_column].dropna())+epochs_to_forecast, freq=frequency_data), name=time_column)
            df_input_short = df_input_short.merge(extended_index, on=time_column, how="right")
                
    ### Launching the model according to chosen methods
    df_outputs_dictionary = {}
    for method in method_parameters_inputs.keys():
        print(f"\nMETHOD USED: {method}\n")
        ### XGBOOST
        if method.upper() == "XGB": 
            ### Processing macro-parameters in XG hyperparameters dictionary
            if "max_previous_steps" in method_parameters_inputs[method].keys():
                if isinstance(method_parameters_inputs[method].get("max_previous_steps"),list):
                    max_previous_steps = method_parameters_inputs[method].get("max_previous_steps")
                else:
                    max_previous_steps = [method_parameters_inputs[method].get("max_previous_steps")]
                del method_parameters_inputs[method]["max_previous_steps"]
            else: 
                max_previous_steps = [1]
            if "n_folds_cv" in method_parameters_inputs[method].keys():
                n_folds_cv = method_parameters_inputs[method].get("n_folds_cv")
                del method_parameters_inputs[method]["n_folds_cv"]
            else:
                n_folds_cv = 4
            if "time_window" in method_parameters_inputs[method].keys():
                if isinstance(method_parameters_inputs[method].get("time_window"),list):
                    time_window = [str(elem)+frequency_data for elem in method_parameters_inputs[method].get("time_window")]
                else:
                    time_window = [str(method_parameters_inputs[method].get("time_window"))+frequency_data]
                del method_parameters_inputs[method]["time_window"]
            else:
                time_window = ["24h"]
            if "aggregation_methods" in method_parameters_inputs[method].keys():
                if  isinstance(method_parameters_inputs[method].get("aggregation_methods"), str):
                    aggregation_methods = [[method_parameters_inputs[method].get("aggregation_methods")]]
                else:
                    aggregation_methods = method_parameters_inputs[method].get("aggregation_methods")
                    for pos in range(len(aggregation_methods)):
                        if isinstance(aggregation_methods[pos], str):
                            aggregation_methods[pos] = [aggregation_methods[pos]]      
                del method_parameters_inputs[method]["aggregation_methods"]
            else:
                aggregation_methods = [["mean"]]
            ### Finding best XGBoost
            from forecast_xgboost import XGBoost_optimization
            start_time = time() 
            df_outputs_dictionary[method] = XGBoost_optimization(df_input=df_input_short, 
                                                                  target_column=target_column, 
                                                                  time_column=time_column, 
                                                                  frequency_data=frequency_data,
                                                                  xgb_params_for_cv=method_parameters_inputs[method], 
                                                                  epochs_to_forecast=epochs_to_forecast, 
                                                                  epochs_to_test=epochs_to_test, 
                                                                  max_previous_steps=max_previous_steps,
                                                                  time_window=time_window,
                                                                  aggregation_methods=aggregation_methods,
                                                                  n_folds_cv=n_folds_cv)
            
            print(f"XGB Computation time: {time()-start_time} sec")
        
        ### LSTM
        if method.upper() == "LSTM": 
            from forecast_lstm import optimised_lstm
            if epochs_to_test == 0:
                warn("Epochs_to_test=0 is not handled by LSTM. Setting train='all' and epochs_to_test=1 instead")
                method_parameters_inputs[method]["train"] = "all"
                epochs_to_test = 1
            start_time = time()
            df_outputs_dictionary[method] = optimised_lstm(df_input_short, 
                                                          target_column=target_column,
                                                          time_column=time_column,
                                                            epochs_to_forecast = epochs_to_forecast,
                                                            epochs_to_test=epochs_to_test, 
                                                            **method_parameters_inputs[method],
                                                            frequency_data = frequency_data)
            
            print(f"LSTM Computation time: {time()-start_time} sec")
       
        ### ARIMAX
        if method.upper() == "SARIMAX": 
            from forecast_arimax import pdm_auto_arima
            start_time = time()
            df_outputs_dictionary[method] = pdm_auto_arima(df_input_short,
                                                          target_column=target_column,
                                                          time_column = time_column,
                                                          epochs_to_forecast = epochs_to_forecast, 
                                                          epochs_to_test=epochs_to_test, 
                                                          frequency_data=frequency_data,
                                                          **method_parameters_inputs[method])
            print(f"SARIMAX Computation time: {time()-start_time} sec")
            
        if method.upper() == "OLS": 
            if len(external_features)==0:
                print("\nWARNING: OLS does not accept dataframes with only time and target. Needing exogenous features. Avoinding this model...\n")
            else:
                from forecast_ols_ar_exp import OLSRegression
                start_time = time()
                df_outputs_dictionary[method] = OLSRegression(df_input=df_input_short, 
                                              target_column=target_column, 
                                              time_column=time_column, 
                                              epochs_to_forecast=epochs_to_forecast, 
                                              epochs_to_test=epochs_to_test)
                print(f"OLS Computation time: {time()-start_time} sec")
            
        if method.upper() == "AUTOREGRESSIVE": 
            from forecast_ols_ar_exp import AutoRegression
            start_time = time()
            df_outputs_dictionary[method] = AutoRegression(df_input=df_input_short, 
                                          target_column=target_column, 
                                          time_column=time_column, 
                                          epochs_to_forecast=epochs_to_forecast, 
                                          epochs_to_test=epochs_to_test,
                                          hyper_params_ar=method_parameters_inputs[method])
            print(f"Autoregressive computation time: {time()-start_time} sec")
            
        if method.upper() == "EXP_SMOOTH": 
            ### OLD MODEL NOT WELL PERFORMING, IT CAN BE DELETED
            from forecast_ols_ar_exp import ExponentialSmoothingRegression
            start_time = time()
            df_outputs_dictionary[method] = ExponentialSmoothingRegression(df_input=df_input_short, 
                                          target_column=target_column, 
                                          time_column=time_column, 
                                          frequency_data=frequency_data,
                                          epochs_to_forecast=epochs_to_forecast, 
                                          epochs_to_test=epochs_to_test,
                                          smoothing_params=method_parameters_inputs[method])
            print(f"Exponential smoothing computation time: {time()-start_time} sec")
    
    ### Output DataFrame
    if show_minimal_columns:
        for method in df_outputs_dictionary.keys():
            df_outputs_dictionary[method] = df_outputs_dictionary[method][columns_to_show]
    
    return df_outputs_dictionary