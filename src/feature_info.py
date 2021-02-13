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
from scipy.signal import correlate
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

#%% GENERAL CALL

def feature_info(df_input, 
                 target_column,
                 time_column):
    """
    This function plots the cross-correlations between target and other features, so as it computes the mutual information.
    Parameters:
        - df_input (pandas.DataFrame): Input Time Series with NaNs in the target column in the time steps to forecast.
        - target_column (str): name of the target feature to be predicted
        - time_column (str): name of the column containing the pandas Timestamps
    Returns:
        - df_MI (pandas.DataFrame): Mutual information score
        - Moreover, it plots the crosscorrelations
    """
       
    
    external_features = [col for col in df_input.columns if col not in [time_column, target_column]]    
    df_temp = df_input.dropna()
    df_MI = {}
   
    ### CROSS-CORRELATION TEST
    if len(external_features)>0:
        df_MI[f"{target_column} (target)"] = mutual_info_regression(df_temp[external_features].to_numpy(), df_temp[target_column].to_numpy())
        for col_exo in external_features+[target_column]:
            cross_corr = correlate(df_temp[col_exo], df_temp[target_column], mode='full', method='auto')
            lag_position = cross_corr.argmax()
            plt.figure()
            plt.plot(np.array(range(-len(df_temp)+1,len(df_temp))), cross_corr)
            if col_exo != target_column:
                plt.title(f"Cross-correlation plot - {target_column} and {col_exo}")
            else:
                plt.title(f"Autocorrelation plot of {target_column}")
        print("\n")
        if len(external_features)>1:
            for pos1 in range(len(external_features)):
                df_MI[external_features[pos1]] = mutual_info_regression(df_temp[external_features].to_numpy(), df_temp[external_features[pos1]].to_numpy().ravel())
        
        df_MI = pd.DataFrame.from_dict(df_MI, orient="index", columns=external_features)
        print("Mutual information scores:")
        print(df_MI)
        plt.show()
        return df_MI     
        
    return pd.DataFrame()
