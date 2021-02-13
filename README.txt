TIME SERIES FORECASTING FRAMEWORK DOCUMENTATION

This document aims to help the user of the framework of Time Series Forecasting (TSFor shortly). This framework is split in 6 files (in "src"):

- "main.py": the main file where we should set file locations and model parameters and run it
- "forecast.py": this file makes a first data processing and it calls the models chosen
- "forecast_arimax.py": this file calls the SARIMAX forecasting, setting parameters through pyramid.auto_arima
- "forecast_lstm.py": this file calls the LSTM forecasting, including the parameter optimization procedure
- "forecast_xgboost.py": this file calls the XGBoost forecasting, including the parameter optimization procedure
- "forecast_ols_ar_exp.py": this file calls the Autoregressive model, Linear regression and the Exponential smoothing model
- "feature_info.py": this file plots the cross-correlation between features and it computes the Mutual Information matrix

Deep-diving in each file:

--- main.py --------------------------------------

This file is split in the four blocks:

1) LIBRARIES, FUNCTIONS AND COSTANTS: here we add also "forecast.py"

2) PARAMETERS SETTING:
This is the block of code where we need to act. Variables to be set are:

-input_filename: the name of the file used as input. The file MUST have the following criteria:
    	- Expected a fixed frequency (daily, weekly, monthly) to be set in "frequency_data" parameter
    	- It must contain only one time column and one target column. Other columns are external features.
    	- Being a ".csv" file, where separator must be ",", new rows are "\n", decimal numbers use the full point "." for decimal part.
-target_column: the string specifying the feature used as target
-time_column : the string specifying the feature used as time column
-frequency_data: an offset ("D","s","M","MS", etc.) where we define what is the frequence of the records. Also multiples ("2D","10ns", etc.) are accepted
-columns_to_exclude: list of names of columns that are in the input dataset but actually we do not want to consider.
-outname: name of output file.
-steps_to_forecast: number of steps to be forecasted in future.
-epochs_to_test: number of steps to be forecasted overlapping the most recent record in order to compute error measures. 
-methods: list of names of the method we want to use. Currently the names used are "XGB","LSTM","SARIMAX","AUTOREGRESSIVE","EXP_SMOOTH" and "OLS"
-method_parameters_inputs[method]: a python dictionary where for each method there are the hyperparameters to apply to the model, most included as a list for optimization purpose (only for LSTM and XGBoost). The keys are the the parameter names used in the model code (as in documentation), the values are the values applie or the list of possible values that should be tested.

3) LOAD AND PROCESS DATA AND OUTPUT
Here the input dataset is uploaded, we do a basic processing and we plot the target feature.

4) GENERIC TIME SERIES FORECASTING
First, we call the "feature_info" function in order to plot cross-correlations (useful for searching the lags and correlations)
between features, and to compute the "mutual information" in order to see the non-linear (but immediate) effects between features.

Then, we call "time_series_forecasting" from "forecast.py". The input are the input dataframe, time and target column names, the dictionary of methods with hyperparameters. The output is a dictionary where key is method name and values id output dataframe.
Finally we compute error metrics for all output dataframes and we plot  the target history and the forecast. Finally we save the output for each method.

--- forecast.py --------------------------------------

This file first calls the libraries where models are defined, i.e.

from forecast_arimax import pdm_auto_arima
from forecast_lstm import optimised_lstm
from forecast_xgboost import XGBoost_optimization
from forecast_ols_ar_exp import OLSRegression, AutoRegression, ExponentialSmoothingRegression_holt, ExponentialSmoothingRegression_simple

Then, the function "time_series_forecasting" is defined. This function process the input dataset and the hyper-parameters dictionary according to the model chosen. Indeed any model has some differences requiring different inputs.
First part of "time_series_forecasting" are compliance checks. We see if epochs_to_test and steps_to_forecast are meaningful and we see if steps_to_forecast goes beyond the last external features available (i.e. we have actual predictors to use). 
Then, we call the forecasting function for each model, but we here describe what are the processing steps before computing the forecast.

- XGB:
This is the most complicated model for processing. The main problem of XGB hyper-parameters dictionary is that we included in the dictionary four parameters that are not actually model hyper-parameters, but feature augmentation ones. These parameters are:
- aggregation_methods
- max_previous_steps
- time_window
- n_folds_cv
We refer to "forecast_xgb.py" chapter for understanding the meaning of these parameters. We split these parameters from the hyper-parameters dictionary and they are used as input in "XGBoost_optimization" call.

- LSTM: Very straightforward, but epochs_to_test must be at least 1
- SARIMAX and OLS: It is not computed if there are no external features.
- EXP_SMOOTH and AUTOREGRESSIVE: Nothing necessary

if show_minimal_column is True, then only basic columns are shown: time, target and forecast.
Finally, the output is a pandas DataFrame where keys are methods and values are the output dataframes.

--- forecast_lstm.py --------------------------------------

This file has two main functions, "lstm" and "optimised_lstm".
Support functions "normalize","normalize_with","matrix_preparation_lstm" and "prepare_data_for_lstm" are used for tensor building.

Function "lstm":
This is actually the core part where the input dataframe is turned into a tensor, the Tensorflow (TF) model is built and trained and forecasts are computed. Calling "prepare_data_for_lstm" we get X and y, that are successively split in train, test and std (test for confidence intervals).
We build the LSTM model according to the input, using n_layers, n_epochs, n_steps, etc., we train it using n_epochs, using X_test and y_test as validation data for evaluating loss decrease and actuating EarlyStoppingCallback for speeding up training.
The variable "forecast" is at first the model predictions on X_test, then forecast is augmented in the future with a "for" loop, adding one record each time. Finally "forecast_df" is built and it is returned.

Function "optimised_lstm":
This function actually performs a grid search of the best parameters in the parameter lists provided as input. After computing the cartesian product of input lists, for each combination it runs function "lstm" and then the "mean absolute percentage error" (MAPE) metrics is evaluated on the test set (i.e. the last epochs_to_test records). The model with the lowest error is kept. Finally, the best model is returned.


--- forecast_arimax.py --------------------------------------

In this file the only function called is "pdm_auto_arima". This file searches for the best SARIMAX model according to the input parameters and then it computes the output.
First, there is some data processing where we observe if there are external features to use as exogenous, we scale external data and we set a fixed frequency.
Then, there is a "if validate: ... else: ..." clause where we do parameter search. If validate is True, we extract the last epochs_to_test rows in order to have a validation set and doing a parameter search where finding the best model (i.e. finding orders p, q, P, Q, etc. of SARIMAX) and we get forecast for the test rows. If validate is False, predictions are not done on the test rows.
We extract variable "exog_forecast" that is used to compute variable "forecast" and "forecast_ci", which are the predictions. In "forecast_df" we build the ourput dataframe, that is returned.


--- forecast_xgboost.py --------------------------------------

This file trains multiple XGBoost models and return the best output found according to input parameters. In particular, XGBoost is not a model thought for time series, so there is an augmentation function which creates past-based features which can help the model.

Function "feature_augmentation":
This function gets the input dataset, time column name and data frequency and three augmentation parameters: "n_previous_steps", "time_window" and "aggregation_methods". The idea is that we compute the moving average (but not necessarily the average, depending on "aggregation_methods") of each feature in a "time_window", going back until "n_previous_steps"
- "aggregation_methods" contains lists including the name of the aggregation type used in pandas "agg" function (e.g. mean, std, median, sum, etc.). We can use also multiple aggregation methods in a list or tuple; anyway even using just one aggregation method, it should be in a list.
- "time_window" is the window where we actually compute the average. Even if it should be a Timedelta object, it is an integer, since it is simply multiplied by the frequency data given in input
-"n_previous_steps" is an integer describing how many time windows in past we have to compute.

EXAMPLE:
With frequency_data as "D" (daily) and feature "temperature", parameters are: aggregation_methods=["mean"], time_window=4 and n_previous_steps=3 --- 
--- What happens is that we add an extra column where it is computed the mean of the temperature in the last 4 days, then we add a second column with the average tempetature on 4 to 8 days before, finally the average on 8 to 12 day before, so 3 added features.

Function "XGBoostRegression":
After feature augmentation, this function perform a Time series crossvalidation on the model hyperparameters provided, and after finding best parameters, it returns the model trained on the best parameters found.

Function "XGBoost_optimization":
This function returns the best model found wih both the model and the feature augmentation parameters found. After finding the best model overall, it makes predictions and it returns the output table with forecasts.


--- forecast_ols_ar_exp.py --------------------------------------
This file makes forecasts based on simpler models: OLS, Autoregressive and exponential smoothing.

Function "OLSRegression":
This function makes a forecast based on linear regression. It needs external features to run. It returns the output forecast.

Function "AutoRegression":
This function makes a forecast based on autoregression, with several parameters including max lag. It returns the output forecast.

Function "ExponentialSmoothingRegression":
This function makes a forecast based on exponential smoothing. It returns the output forecast.

--- feature_info.py --------------------------------------

this file plots  cross-correlation between features and target, which is useful for lag detection, and it computes the mutual information between features and target, which is useful for feature selection.