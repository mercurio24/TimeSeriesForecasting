# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 2020

@author: Fabio Mercurio

This program aims to be a generalized API to do Time Series Forecasting using SARIMAX, LSTM or XGBoost
    
DATASETS REQUIRE THE FOLLOWING PREPROCESSING STEPS:
    - Expected a fixed frequency (daily, weekly, monthly) to be set in "frequency_data" parameter
    - It must contain only one time column and one target column. Other columns are input
    - Reading .csv files as input. Separator must be ",", new rows are "\n", decimal numbers use the full point "." for decimal part.
    - ARIMAX and OLS cannot perform if there are no external features.
"""
#%% LIBRARIES, FUNCTIONS AND COSTANTS

import os#, math
import numpy as np
import pandas as pd
from forecast import time_series_forecasting
from feature_info import feature_info
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
def MAPE_MAE_RMSE(df_output):
    if len(df_output[(~df_output[target_column].isna())&(~df_output["forecast"].isna())])>0:
        df_error_check = df_output[(~df_output[target_column].isna())&(~df_output["forecast"].isna())]
        MAPE = mean_absolute_percentage_error(df_error_check[target_column].values, df_error_check["forecast"].values)
        MAE = mean_absolute_error(df_error_check[target_column].values, df_error_check["forecast"].values)
        RMSE = np.sqrt(mean_squared_error(df_error_check[target_column].values, df_error_check["forecast"].values))
        return MAPE, MAE, RMSE
    else:
        return np.inf, np.inf, np.inf
    
#%% PARAMETERS SETTING

### Input, exogenous and output DATA SETTINGS
input_filename = "barilla_ICA_CRISPBREAD_Wasa.csv" # barilla_ICA_CRISPBREAD_Wasa, covid_italy_cases_processed , AirQualityUCI_processed , delhi_climate_temperature_processed
target_column = "Amount" #Amount, New_cases , NOx(GT) , meantemp ### MANDATORY TARGET COLUMN TO DEFINE
time_column = "data" # data, Date_reported, Time , date  ### MANDATORY TARGET COLUMN(S) TO DEFINE
frequency_data = "MS" # Offset
columns_to_exclude = [] #"New_deaths" ### Here you can list the columns that must not be considered in input and exogenous

outname = input_filename[:-4]+ "_forecast.csv" #Forecast filename

#GENERIC FORECAST PARAMETERS
max_epochs_to_forecast = None   #Forecast this amount of epochs
epochs_to_test     = 3  #Test the forecasting on these last ephocs, set 0 if no validation is wanted

### FORECAST SETTINGS
### I get errors for no exogenous with OLS and SARIMAX
methods = ["LSTM","SARIMAX","XGB","AUTOREGRESSIVE","OLS"] # Forecasting model chosen

method_parameters_inputs = {}

for method in methods:
    if method.upper() == "SARIMAX": 
        ### Hyper-parameters to be chosen for ARIMAX
        ### TO-DO: ADD EXTERNAL FEATURES INDEPENDENCE ON SARIMAX
        method_parameters_inputs[method] = {"d":1,
                                             "D":0,
                                             "seasonal":False,
                                             "m":12,
                                             "start_p":1,
                                             "max_p":2,
                                             "start_q":1,
                                             "max_q":2,
                                             "start_P":1,
                                             "max_P":2,
                                             "start_Q":1,
                                             "max_Q":2,
                                             "validate":True}
        
    if method.upper() == "XGB": 
        ### Hyper-parameters to be chosen for XGBoost
        method_parameters_inputs[method] = {"n_estimators":[10],
                                             "learning_rate":[0.3,0.1],
                                             "booster":["gbtree"],
                                             "max_depth":[4,8],
                                             "gamma":[0,1],
                                             "min_child_weight":[0,10],
                                             "subsample":[0.9],
                                             "colsample_bytree":[0.9],
                                             "colsample_bylevel":[1],
                                             "colsample_bynode":[1],
                                             "reg_alpha":[0],
                                             "reg_lambda":[0.1],
                                             "max_previous_steps":[3],
                                             "time_window":[3],
                                             "aggregation_methods":["median",("mean","std")]}

    if method.upper() == "LSTM": 
        ### Hyper-parameters to be chosen for LSTM
        method_parameters_inputs[method] = {"epochs_for_std":1,
                                            "n_steps":[5],
                                            "n_layers":[2,4],
                                            "n_neurons":[20],
                                            "n_epochs":[300],
                                            "patience":60,
                                            "batch_size":None,
                                            "train":"none"}
        
    if method.upper() == "OLS":
        method_parameters_inputs[method] = {}
        
    if method.upper() == "AUTOREGRESSIVE":
        method_parameters_inputs[method] = {"maxlag": 10, 
                                           "trend":'c', 
                                           "ic": "bic",
                                           "seasonal":True, 
                                           "period":None, 
                                           "glob" : False,
                                           "missing":'none'}
    
    # EXPONENTIAL SMOOTHING NOT CONSIDERED
    if method.upper() == "EXP_SMOOTH":
        method_parameters_inputs[method] = {"span":15,
                                            "adjust":True}
        #method_parameters_inputs[method] = {}

#%% LOAD AND PROCESS DATA AND OUTPUT

df_input = pd.read_csv(os.path.join("..","data",input_filename)).dropna(axis=1, how='all').dropna(axis=0, how='all')
df_input[time_column] = pd.to_datetime(df_input[time_column])
df_input = df_input[[elem for elem in df_input.columns if elem not in columns_to_exclude]]

#Setting forecast steps if they are not defined
if max_epochs_to_forecast is None or not isinstance(max_epochs_to_forecast, int):
    last_non_null_time = max(df_input[~df_input[target_column].isna()][time_column])
    max_epochs_to_forecast = len(df_input[df_input[time_column]>last_non_null_time])

### Target plot
plt.figure(figsize=(14,7))
df_input.set_index(time_column, 1)[target_column].plot()
plt.title("Target feature")
plt.xlabel(time_column)
plt.ylabel(target_column)
plt.show()

#%% GENERIC TIME SERIES FORECASTING

### Cross-correlations and mutual information
df_MI = feature_info(df_input, target_column, time_column)
    
### Time series forecasting
df_outputs_dictionary = time_series_forecasting(df_input=df_input, 
                                                target_column=target_column,
                                                time_column=time_column,
                                                frequency_data=frequency_data,
                                                epochs_to_forecast=max_epochs_to_forecast,
                                                epochs_to_test=epochs_to_test,
                                                method_parameters_inputs=method_parameters_inputs)

### Computing error and plotting reults
for method in df_outputs_dictionary.keys():
    
    MAPE, MAE, RMSE = MAPE_MAE_RMSE(df_outputs_dictionary[method])
    print(f"For method {method}:\nMAPE = {MAPE}%\nMAE = {MAE}\nRMSE = {RMSE}\n")
    
    plt.figure(figsize=(15,7.5))
    df_outputs_dictionary[method].set_index(time_column)[target_column].plot(color="tab:red")
    df_outputs_dictionary[method].set_index(time_column)["forecast"].plot(color="tab:blue",linewidth=2)
    plt.title(f"{method} prediction - MAPE={round(MAPE,2)}%")
    plt.savefig(os.path.join("..","plot",f"{input_filename[:-4]}_{method}.png"),dpi=200)
    
    ### Saving
    df_outputs_dictionary[method].to_csv(os.path.join("..","out",f"{method}_{outname}"), index=False)
    
plt.show()