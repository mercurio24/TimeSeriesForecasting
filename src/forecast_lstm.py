'''
Project: LSTM generic Time Series Forecasting

Contributors:
Marocchino Alberto
Daniele Iandolo
Fabio Mercurio
'''

#%% LIBRARIES AND SETTINGS

import os
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import tensorflow as tf # last version used here 2.1.0
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.python.keras.models  import Sequential
from tensorflow.python.keras.layers  import GRU, Dense, LSTM, Bidirectional,Dropout
from tensorflow.python.keras         import callbacks
from tensorflow.python.keras         import optimizers
from tensorflow.python.keras.callbacks import EarlyStopping

tf.config.experimental.list_physical_devices('GPU') 

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
   
def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
#%% Core part of LSTM training, where Tensorflow is called

# Prepare data
def normalize(df):
    normalization = []
    for idx in range(df.shape[1]):
        minimum, maximum = df.iloc[:,idx].min(), df.iloc[:,idx].max()
        df.iloc[:,idx] = (df.iloc[:,idx]-minimum) / (maximum-minimum)
        normalization.append( [minimum,maximum] )
    return df, normalization

def normalize_with(df,normalization):
    for idx in range(df.shape[1]):
        minimum, maximum = normalization[idx][0], normalization[idx][1]
        df.iloc[:,idx] = (df.iloc[:,idx]-minimum) / (maximum-minimum)
    return df

def matrix_preparation_lstm(df, n_steps):
    X, y = list(), list()
    for i in range(len(df)):
        #find where to stop the loop
        end_idx = i + n_steps
        #check if you are beyond the sequence
        if end_idx > len(df)-1:
            break
        #get data from the vector and put them in the new matrix
        seq_y = df.iloc[end_idx, 0]
        seq_x = []
        for j in range(df.shape[1]):
            seq_x.extend(df.iloc[i:end_idx, j])
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

      
def prepare_data_for_lstm(df, n_steps, normalization=None):
    if df.shape[1]>1:
        print('the dataframe has more than a raw, multidimensional approach')
    else:
        print('the dataframe has one single column, one dimensional approach')
    if normalization==None:
        print('this is the first normalization, coefficients are calculated')
        df, normalization = normalize(df)
    else:
        print('existing normalization, adapting')
        df = normalize_with(df,normalization)
    X, y = matrix_preparation_lstm(df, n_steps)
    return X, y, normalization


def lstm(df_input, target_column,
                  epochs_to_forecast = 1,
                  epochs_to_test     = 1,
                  epochs_for_std     = 1,
                  n_steps            = 2,
                  n_feautures        = 1,
                  n_layers           = 1,
                  n_neurons          = 10,
                  n_epochs           = 10,
                  train              = 'none',
                  patience           = 100,
                  batch_size         = 10):

    #%% prepare dataset for lstm
    df=df_input.dropna()
    X, y, normalization = prepare_data_for_lstm(df.copy(), n_steps)
    n_step_multiplier   = df.shape[1]

    if train=='all':
        X_train, y_train = X[:, :], y[:]
        X_test,  y_test  = X[-epochs_to_test:, :], y[-epochs_to_test:]
        X_std,   y_std   = X[-epochs_for_std:, :], y[-epochs_for_std:]
    else:
        X_train, y_train = X[:-epochs_to_test, :], y[:-epochs_to_test]
        X_test,  y_test  = X[-epochs_to_test:, :], y[-epochs_to_test:]
        X_std,   y_std   = X[-epochs_to_test-epochs_for_std:-epochs_to_test,:], y[-epochs_to_test-epochs_for_std:-epochs_to_test]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_feautures))
    X_test  = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1],  n_feautures))
    X_std   = np.reshape(X_std,   (X_std.shape[0],   X_std.shape[1],   n_feautures))

    #%% lstm
    print("SHAPE ", X_train.shape)

    print("Initializing learning phase")
    model       = Sequential()
    layer_follow = False if n_layers<=1 else True
    model.add(LSTM(n_neurons, activation='relu',kernel_initializer="he_uniform", return_sequences=layer_follow, input_shape=(n_steps*n_step_multiplier,n_feautures)))
    for idx in np.arange(1,n_layers):
        layer_follow = False if idx == n_layers-1 else True
        model.add(LSTM(n_neurons, activation='relu',kernel_initializer="he_uniform", return_sequences=layer_follow))
    #adding dropout layer
    model.add( Dropout(0.1))
    model.add( Dense(1) )
    
    #prima era mse
    model.compile(optimizer='Adam', loss='mse', metrics=['mse'])

    #ADDING EARLY STOPPING IN ORDER TO AVOID OVERFITTING AND RUNNING FOR TOO LONG 
    earlystop_callback = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.001,patience=patience) 
    history = model.fit(X_train, y_train, validation_data=(X_test,y_test), 
              epochs=n_epochs, verbose=1, batch_size=batch_size, callbacks=[earlystop_callback]) #batch size = n_features vuol dire che n_features è sempre 1. Per questo sta una vita!
    
    #%% forecast
    y_train = y_train * (normalization[0][1] - normalization[0][0]) + normalization[0][0]
    y_test  = y_test  * (normalization[0][1] - normalization[0][0]) + normalization[0][0]
    y_std   = y_std   * (normalization[0][1] - normalization[0][0]) + normalization[0][0]
    
    forecast_std = model.predict(X_std)  * (normalization[0][1]-normalization[0][0]) + normalization[0][0]
    σ            = np.std(y_std-forecast_std[:,0])

    forecast        = model.predict(X_test) * (normalization[0][1]-normalization[0][0]) + normalization[0][0]
    forecast_vector = np.concatenate( forecast )
    df_temp         = df_input.copy()
    for idx in range(epochs_to_forecast):
        df_temp = df_temp.fillna(forecast[-1][0],limit=1)
        df = df_temp.dropna()
        X, y, normalization = prepare_data_for_lstm(df.copy(), n_steps, normalization=normalization)
        n_step_multiplier   = df.shape[1]

        X_test,  y_test  = X[-1:, :], y[-1:]
        X_test    = np.reshape(X_test,  (X_test.shape[0],  X_test.shape[1],  n_feautures))

        forecast  = model.predict(X_test) * (normalization[0][1]-normalization[0][0]) + normalization[0][0]
    
        forecast_vector = np.concatenate( (forecast_vector, forecast[-1]) )
    forecast_vector[ forecast_vector<0. ]=0.
    
    #output
    forecast_df=pd.DataFrame()
    forecast_df[target_column] = np.concatenate( (df_input.dropna()[target_column].values,np.zeros(epochs_to_forecast)*np.nan), axis=0)
    forecast_df['forecast']     = np.nan
    forecast_df['forecast_up']  = np.nan
    forecast_df['forecast_low'] = np.nan
    forecast_df['forecast'].iloc[-epochs_to_forecast-epochs_to_test:]     = forecast_vector
    forecast_df['forecast_up'].iloc[-epochs_to_forecast-epochs_to_test:]  = [np.max([x+3*σ,0]) for x in forecast_vector] #forecast_vector + 3*σ
    forecast_df['forecast_low'].iloc[-epochs_to_forecast-epochs_to_test:] = [np.max([x-3*σ,0]) for x in forecast_vector] #forecast_vector - 3*σ
    return forecast_df



##%% Reasearch of best hyperparameters for LSTM
def optimised_lstm(df_input, target_column, time_column, frequency_data,
                      epochs_to_forecast  = 12,
                      epochs_to_test      = 12,
                      epochs_for_std      = 12,
                      n_steps             = np.array([2,3,4]),
                      n_feautures         = 1,
                      n_layers            = np.array([1]),
                      n_neurons           = np.array([70,80,90]),
                      n_epochs            = np.array([50,75,200]),
                      train               ='all',
                      patience = 100,
                      batch_size = 10):
    """
    This function makes a grid seach of the best parameters for LSTM, making and evaluating forecasts for each combination
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series.
        - target_column (str): name of the column containing the target feature
        - time_column (str): name of the column containing the pandas Timestamps
        - frequency_data (str): string representing the time frequency of record, e.g. "h" (hours), "D" (days), "M" (months)
        - epochs_to_forecast (int): number of steps for predicting future data
        - epochs_to_test (int): number of steps corresponding to most recent records to test on
        - epochs_for_std (int): number of steps corresponding to most recent records to test standard deviation
        - n_steps (list): numbers of steps for preparing input tensors in the model
        - n_feautures (list): numbers of features used for preparing input tensors in the model
        - n_layers (list): numbers of layers set to the model
        - n_neurons (list): numbers of neurons for each layer
        - n_epochs (int): number of training epochs
        - train (str): if "all", test data are used for training too
        - patience (int): patience for Tensorflow training
        - batch_size (int): batch_size for Tensorflow training
    Returns:
        - forecast_df (pandas.DataFrame): Output DataFrame with best forecast found
    """
    
    assert isinstance(target_column, str)
    assert isinstance(time_column, str)

    best_fit_distance = 1e30
    best_fit_forecast = pd.DataFrame()
    
    combination_counter = 0
    n_combinations = len(n_steps)*len(n_layers)*len(n_neurons)*len(n_epochs)
    
    for (step, layer, neuron, epoch) in itertools.product(n_steps, n_layers, n_neurons, n_epochs):
        combination_counter += 1
        print(f"TRAINING COMBINATION #{combination_counter} on {n_combinations}")
        #date = pd.date_range(start=df_input[time_column].min(), periods=len(df_input), freq=frequency_data)
        date = df_input[time_column]

        #del df_input['data']
        print('Running the RNN with the following parameters - steps:{}, layers:{}, neurons:{}, epochs:{}, epochs_to_forecast: {}, test: {}'.format(step,layer,neuron,epoch,epochs_to_forecast,epochs_to_test))

        forecast = lstm(df_input.drop([time_column],axis=1), 
                                    target_column=target_column,
                                    epochs_to_forecast = epochs_to_forecast,
                                    epochs_to_test     = epochs_to_test,
                                    epochs_for_std     = epochs_for_std,
                                    n_steps            = step,
                                    n_feautures        = n_feautures,
                                    n_layers           = layer,
                                    n_neurons          = neuron,
                                    n_epochs           = epoch,
                                    train              = train,
                                    patience = patience,
                                    batch_size = batch_size)
        forecast[time_column] = date
        forecast_nonan = forecast.dropna()
        #norma 2
        #dist           = np.sqrt(np.sum((forecast_nonan.Amount.values-forecast_nonan.forecast.values)**2))
        #mean absolute error
        dist = mean_absolute_percentage_error(forecast_nonan[target_column],forecast_nonan.forecast)
        
        #output params on a file         
        paramList = [epochs_to_forecast, epochs_to_test, epochs_for_std, step, n_feautures, layer, neuron, epoch,train,batch_size]  
        
        if dist < best_fit_distance and forecast['forecast'].max()<10*forecast_nonan[target_column].max(): #-- with a top threshold
            best_fit_forecast = forecast.copy()
            best_fit_distance = dist   
                
    #best_fit_forecast[time_column] = df_input[time_column].values
    return best_fit_forecast