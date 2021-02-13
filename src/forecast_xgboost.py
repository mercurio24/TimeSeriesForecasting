'''
Project: Generic Time Series Forecasting

Contributors:
Fabio Mercurio
'''

#%%
import os
import numpy as np
import pandas as pd
import itertools
from warnings import warn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%% XGBOOST But with feature augmentation based on aggregations on time intervals

def feature_augmentation(df_input, 
                         time_column, 
                         frequency_data,
                         max_previous_steps=1,
                         time_window=["24h"],
                         aggregation_methods=["mean"]):
    """
    This function adds features based on aggregation on past records.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - max_previous_steps (int): number of features added representing multiple steps in time
        - time_window (np.timedelta64): time interval where aggregation is performed for each step
        - aggregation_methods (list): list of aggregation methods used (e.g. mean, median, std) 
    Returns:
        - df_processed (pandas.DataFrame): Output DataFrame with added features
    """
    
    df_input_new, df_processed = df_input.copy(), df_input.copy()
    last_available_date = np.datetime64(max(df_input_new[time_column].dropna()))
    exog_columns = [col for col in df_input_new.columns if col!=time_column]
    df_input_new = df_input_new.set_index(time_column)
    for aggr_method in aggregation_methods:
        for col in exog_columns:
            for step in range(1,max_previous_steps+1):
                col_name = f"{col}_{aggr_method}_{str(step)}"
                shifted_col = df_input_new[col].shift(periods=step , freq=time_window)
                if "M" in time_window:
                    num = time_window.replace("M","")
                    num = int(num.replace("S",""))*30
                    shifted_col = shifted_col.rolling(str(num)+"D").agg(aggr_method)
                elif "Y" in time_window:
                    num = time_window.replace("M","")
                    num = int(num.replace("S",""))*365
                    shifted_col = shifted_col.rolling(str(num)+"D").agg(aggr_method)
                elif "W" in time_window:
                    num = time_window.replace("W","")
                    num = int(num.replace("S",""))*7
                    shifted_col = shifted_col.rolling(str(num)+"D").agg(aggr_method)
                else:
                    shifted_col = shifted_col.rolling(time_window).agg(aggr_method)
                shifted_col = pd.Series(shifted_col, name=col_name)
                df_processed = df_processed.merge(shifted_col.reset_index(), on=time_column, how="left")
    return df_processed
    

def XGBoostRegression(df_input, 
                      target_column, 
                      time_column, 
                      frequency_data,
                      xgb_params_for_cv={}, 
                      epochs_to_forecast=1, 
                      epochs_to_test=1, 
                      max_previous_steps=1,
                      time_window=["24h"],
                      aggregation_methods=["mean"],
                      n_folds_cv=4):
    """
    This function performs regression using feature augmentation and then training XGB with Crossvalidation.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - xgb_params_for_cv (dict): hyperparameters combinations for cv
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - max_previous_steps (int): number of features added representing multiple steps in time
        - time_window (np.timedelta64): time interval where aggregation is performed for each step
        - aggregation_methods (list): list of aggregation methods used (e.g. mean, median, std) 
        - n_folds_cv (int): number of folds for the CV
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with forecast
        - best_params_of_grid (dict): best parameters of Grid-search CV
    """
    
    df_input_new = df_input.set_index(time_column).asfreq(freq=frequency_data, method="bfill").reset_index()
    
    ### HERE I ADD FEATURES CONSISTING IN THE AGGREGATION OF PAST TARGET FOR n_previous_steps OF time_window
    df_input_new = feature_augmentation(df_input_new, 
                                        time_column, 
                                        frequency_data,
                                        max_previous_steps,
                                        time_window,
                                        aggregation_methods)
    
    model = XGBRegressor(verbosity=1)

    #Time series Cross-Validation for XGBoost
    count_cv_combs = 0
    for _ in itertools.product(*xgb_params_for_cv.values()):
        count_cv_combs += 1
        
    if count_cv_combs == 1:        
        print("There is a unique combination for CV. Setting it directly as best combination")
        best_params_of_grid = {key:xgb_params_for_cv[key][0] for key in xgb_params_for_cv.keys()}
    else:
        print("Performing Grid search CV")
        grid_cv = GridSearchCV(estimator=model, param_grid=xgb_params_for_cv, cv=TimeSeriesSplit(n_folds_cv), verbose=1)
        grid_cv.fit(df_input_new[: -(epochs_to_forecast+epochs_to_test)].drop([time_column,target_column],1), df_input_new[: -(epochs_to_forecast+epochs_to_test)][target_column])
        #grid_cv.fit(df_input_new[: -(epochs_to_forecast+epochs_to_test)].dropna().drop([time_column,target_column],1), df_input_new[: -(epochs_to_forecast+epochs_to_test)].dropna()[target_column])
        best_params_of_grid = grid_cv.best_params_
    model = XGBRegressor(**best_params_of_grid)
    model.fit(df_input_new[:-(epochs_to_forecast+epochs_to_test)].drop([time_column,target_column],1), df_input_new[:-(epochs_to_forecast+epochs_to_test)][target_column])

    forecast = model.predict(df_input_new.drop([time_column,target_column],1))
    forecast[: -(epochs_to_forecast+epochs_to_test)] = np.nan
    df_output = df_input_new.copy()
    df_output["forecast"] = forecast
    #Bisognerebbe definire una deviazione standard
    df_output["forecast_up"] = forecast * 1.1
    df_output["forecast_low"] = forecast * 0.9
    
    return df_output, best_params_of_grid

#%% XGBOOST Optimizer

def XGBoost_optimization(df_input, 
                      target_column, 
                      time_column, 
                      frequency_data,
                      xgb_params_for_cv={}, 
                      epochs_to_forecast=1, 
                      epochs_to_test=1, 
                      max_previous_steps=[1],
                      time_window=["24h"],
                      aggregation_methods=[["mean"]],
                      n_folds_cv=4):
    """
    This function performs regression using feature augmentation and then training XGB with Crossvalidation.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - xgb_params_for_cv (dict): hyperparameters combinations for cv
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - max_previous_steps (int): number of features added representing multiple steps in time
        - time_window (np.timedelta64): time interval where aggregation is performed for each step
        - aggregation_methods (list): list of aggregation methods used (e.g. mean, median, std) 
        - n_folds_cv (int): number of folds for the CV
    Returns:
        - df_output (pandas.DataFrame): Output DataFrame with forecast
    """
    
    if epochs_to_test == 0:
        warn("epochs_to_test=0 does not permit parameters testing, setting it to 1")
        epochs_to_test = 1
        
    best_MAPE = 1e30
    best_comb = None
    best_cv_params = None
    combination_counter = 0
    n_macro_combinations = len(max_previous_steps)*len(time_window)*len(aggregation_methods)
    
    for (max_steps, window, aggr_method) in itertools.product(max_previous_steps, time_window, aggregation_methods):
        combination_counter += 1
        print(f"TRAINING MACRO-COMBINATION {combination_counter} on {n_macro_combinations}")
        df_out_temp, cv_params = XGBoostRegression(df_input=df_input, 
                                          target_column=target_column, 
                                          time_column=time_column, 
                                          frequency_data=frequency_data,
                                          xgb_params_for_cv=xgb_params_for_cv, 
                                          epochs_to_forecast=epochs_to_forecast, 
                                          epochs_to_test=epochs_to_test, 
                                          max_previous_steps=max_steps,
                                          time_window=window,
                                          aggregation_methods=aggr_method,
                                          n_folds_cv=n_folds_cv)
        
        if len(df_out_temp[(~df_out_temp[target_column].isna())&(~df_out_temp["forecast"].isna())])>0:
            df_error_check = df_out_temp[(~df_out_temp[target_column].isna())&(~df_out_temp["forecast"].isna())]
            MAPE = mean_absolute_percentage_error(df_error_check[target_column].values, df_error_check["forecast"].values)
        if MAPE < best_MAPE:
            best_MAPE = MAPE
            best_comb = (max_steps, window, aggr_method)
            best_cv_params = cv_params
            
            
    print(f"\nBest MAPE is {best_MAPE}\nBest combination optimization for XGBoost is {best_comb} and {best_cv_params}\n")
    max_steps, window, aggr_method = best_comb
    best_cv_params = {elem:[best_cv_params[elem]] for elem in best_cv_params.keys()}
    
    df_output, _ = XGBoostRegression(df_input=df_input, 
                                    target_column=target_column, 
                                    time_column=time_column, 
                                    frequency_data=frequency_data,
                                    xgb_params_for_cv=best_cv_params, 
                                    epochs_to_forecast=epochs_to_forecast, 
                                    epochs_to_test=epochs_to_test, 
                                    max_previous_steps=max_steps,
                                    time_window=window,
                                    aggregation_methods=aggr_method,
                                    n_folds_cv=n_folds_cv)
    return df_output
            
        

    