# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 23:22:38 2024

@author: lucas

This script is the preprocessing of the datasets before GIMME
"""
import pandas as pd
import numpy as np
import sklearn as skl
import os

# Loading the file
df = pd.read_excel("C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/RPE_data.xlsx")

list_names = df['NOME'].tolist()
df.index = list_names  # Setting the indexes

df.replace(0, np.nan, inplace=True)  # Replacing 0 with NaN
df_dropped = df.drop(df.columns[[0, 1, 2, 46, 47]], axis = 1)  # These columns have lots of NaN
df_dropped['TL- 02/10'] = 0

# Iputing the data with KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=2)  # Setting the imputer
RPE_final = imputer.fit_transform(df_dropped)  # Let's impute
df_RPE = pd.DataFrame(RPE_final, index = list_names, columns = df_dropped.columns)

#%% Manipulate the Wellness file
wellness_path = "C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/RPE_data_2.xlsx"

df_W = pd.read_excel(wellness_path, sheet_name = None)
output_folder = "C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/GIMME/TEMPORAL_SERIES DATA"


W_final = {}  # Create a dictionary with all dfs imputed

for sheet_name, df1 in df_W.items():
    df1.replace(0, np.nan, inplace=True)
    df1 = df1.drop('Data', axis = 1)
    rows_to_drop = [43, 44]  # Droppig the rows
    df1.drop(index=rows_to_drop, errors='ignore', inplace=True)
    
    numeric_cols = df1.select_dtypes(include=['number']).columns  # I have to divide the data from the value
    df_numeric = df1[numeric_cols]
    df_imputed = pd.DataFrame(imputer.fit_transform(df_numeric), columns=numeric_cols)  # I have to impute along the columns
    
    W_final[sheet_name] = pd.concat([df1[df1.select_dtypes(exclude=['number']).columns], df_imputed], axis=1)


for sheet_name, df2 in W_final.items():  # I create a different df for every sample
    for idx in df_RPE.index:
        if sheet_name in df_RPE.index:
            df2['TL'] = df_RPE.loc[sheet_name].values

    output_file = os.path.join(output_folder, f'{sheet_name}.csv')  # Saving within-sample time series created
    df2.to_csv(output_file, index=False)


'''
One problem:
    1. check if the W data splitted for sheet are correct
    2. check if the TL are loaded correctly
'''
