# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import random
import seaborn as sns


df = pd.read_excel(
    'C:/Users/lucas/OneDrive - Università degli Studi di Trieste/TRIENNALE/TESI_RICERCA/DATI/FISI/Data_Spyder_python.xlsx')

#%% Cleaning and manipulating the data before the analysis
# The most important dataframes will be: df, df_date_e_cronico, df_date_e_acuto
# df contains all the data
# df_date_e_cronico contains all the cronico that are matched with RESTQ's days
# df_date_e_acuto contains all the cronico that are matched with RESTQ's days
# df_W contains all of the W's values that are matched with RESTQ's days
# df_RESTQ contains all of the RESTQ's values of all the measures

# Copy the df to do the Acuto and cronico analysis
df2 = df.copy()

# dropping the TL and W of the days (don't need)
df = df.drop(df.columns[2:90], axis = 1)
# Dropping a sample with too many misses
df = df[df['Partecipanti'] != 'Skier-HA']

# I want to find the location of the last columns
x = df.columns.get_loc('date1')
y = df.columns.get_loc('date6')

# Transform this columns in strings
df_date = df.iloc[:, 138:144]  # Creating a df only for dates
df_date_string = df_date.map(str)
df_date_string.dtypes  # The values are strings

# I cut the values splicing year and time and replacing - with /
df_date_string['estratti1'] = df_date_string['date1'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )
df_date_string['estratti2'] = df_date_string['date2'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )
df_date_string['estratti3'] = df_date_string['date3'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )
df_date_string['estratti4'] = df_date_string['date4'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )
df_date_string['estratti5'] = df_date_string['date5'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )
df_date_string['estratti6'] = df_date_string['date6'].apply(
    lambda x: x.replace('-', '/')[5:10]
    )

# Creating a clean df with only the useful date's format
df_date_final = df_date_string.loc[:, 'estratti1':'estratti6']
df_date_final.columns = [
    'Date1', 'Date2', 'Date3', 'Date4', 'Date5', 'Date6'
    ]
# I need to turn the moth-day formati into day-month
def turn_date(data):
    month, day = data.split('/')
    return f"{day}/{month}"

df_date_final = df_date_final.loc[:, 'Date1':'Date6'].map(turn_date)

# Extract a df only for "Acuti" and "Cronico"
df_acuti = df.filter(like='Acuto-')
df_cronico = df.filter(like='Cronico-')


# Giving the name of the df indexes to all dfs
z = list(df['Partecipanti'].astype(str))
df.index = df_acuti.index = df_cronico.index = df_date_final.index = z
del df['Partecipanti']  # Removing the extra columns called 'Partecipanti'

# I'll filter the values of the single samples for cronico

# DW
df_date_DW = df_date_final.loc['Skier-DW', :]
df_acuti_DW = df_acuti.loc['Skier-DW', :]

# Defining the df of every sample
risultati_DW = pd.DataFrame()

for i in range(6):
    g = df_date_DW.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_DW.index:
        risultati_DW = pd.concat([risultati_DW, df_acuti_DW.loc[[acuto_str]]])
    else: print("none")

# TG
df_date_TG = df_date_final.loc['Skier-TG', :]
df_acuti_TG = df_acuti.loc['Skier-TG', :]

# Defining the df of every sample
risultati_TG = pd.DataFrame()

for i in range(6):
    g = df_date_TG.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_TG.index:
        risultati_TG = pd.concat([risultati_TG, df_acuti_TG.loc[[acuto_str]]])
    else: print("none")

# DB
df_date_DB = df_date_final.loc['Skier-DB', :]
df_acuti_DB = df_acuti.loc['Skier-DB', :]

# Defining the df of every sample
risultati_DB = pd.DataFrame()

for i in range(6):
    g = df_date_DB.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_DB.index:
        risultati_DB = pd.concat([risultati_DB, df_acuti_DB.loc[[acuto_str]]])
    else: print("none")

# EZ
df_date_EZ = df_date_final.loc['Skier-EZ', :]
df_acuti_EZ = df_acuti.loc['Skier-EZ', :]

# Defining the df of every sample
risultati_EZ = pd.DataFrame()

for i in range(6):
    g = df_date_EZ.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_EZ.index:
        risultati_EZ = pd.concat([risultati_EZ, df_acuti_EZ.loc[[acuto_str]]])
    else: print("none")

# PB
df_date_PB = df_date_final.loc['Skier-PB', :]
df_acuti_PB = df_acuti.loc['Skier-PB', :]

# Defining the df of every sample
risultati_PB = pd.DataFrame()

for i in range(6):
    g = df_date_PB.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_PB.index:
        risultati_PB = pd.concat([risultati_PB, df_acuti_PB.loc[[acuto_str]]])
    else: print("none")

# DC
df_date_DC = df_date_final.loc['Skier-DC', :]
df_acuti_DC = df_acuti.loc['Skier-DC', :]

# Defining the df of every sample
risultati_DC = pd.DataFrame()

for i in range(6):
    g = df_date_DC.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_DC.index:
        risultati_DC = pd.concat([risultati_DC, df_acuti_DC.loc[[acuto_str]]])
    else: print("none")

# MM
df_date_MM = df_date_final.loc['Skier-MM', :]
df_acuti_MM = df_acuti.loc['Skier-MM', :]

# Defining the df of every sample
risultati_MM = pd.DataFrame()

for i in range(6):
    g = df_date_MM.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_DC.index:
        risultati_MM = pd.concat([risultati_MM, df_acuti_DC.loc[[acuto_str]]])
    else: print("none")

# FD
df_date_FD = df_date_final.loc['Skier-FD', :]
df_acuti_FD = df_acuti.loc['Skier-FD', :]

# Defining the df of every sample
risultati_FD = pd.DataFrame()

for i in range(6):
    g = df_date_FD.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_FD.index:
        risultati_FD = pd.concat([risultati_FD, df_acuti_FD.loc[[acuto_str]]])
    else: print("none")

# CC
df_date_CC = df_date_final.loc['Skier-CC', :]
df_acuti_CC = df_acuti.loc['Skier-CC', :]

# Defining the df of every sample
risultati_CC = pd.DataFrame()

for i in range(6):
    g = df_date_CC.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_CC.index:
        risultati_CC = pd.concat([risultati_CC, df_acuti_CC.loc[[acuto_str]]])
    else: print("none")

# IL
df_date_IL = df_date_final.loc['skier-IL', :]
df_acuti_IL = df_acuti.loc['skier-IL', :]

# Defining the df of every sample
risultati_IL = pd.DataFrame()

for i in range(6):
    g = df_date_IL.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_IL.index:
        risultati_IL = pd.concat([risultati_IL, df_acuti_IL.loc[[acuto_str]]])
    else: print("none")

# DZ
df_date_DZ = df_date_final.loc['Skier-DZ', :]
df_acuti_DZ = df_acuti.loc['Skier-DZ', :]

# Defining the df of every sample
risultati_DZ = pd.DataFrame()

for i in range(6):
    g = df_date_DZ.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_DZ.index:
        risultati_DZ = pd.concat([risultati_DZ, df_acuti_DZ.loc[[acuto_str]]])
    else: print("none")

# SS
df_date_SS = df_date_final.loc['Skier-SS', :]
df_acuti_SS = df_acuti.loc['Skier-SS', :]

# Defining the df of every sample
risultati_SS = pd.DataFrame()

for i in range(6):
    g = df_date_SS.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_SS.index:
        risultati_SS = pd.concat([risultati_SS, df_acuti_SS.loc[[acuto_str]]])
    else: print("none")

# MT
df_date_MT = df_date_final.loc['Skier-MT', :]
df_acuti_MT = df_acuti.loc['Skier-MT', :]

# Defining the df of every sample
risultati_MT = pd.DataFrame()

for i in range(6):
    g = df_date_MT.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_MT.index:
        risultati_MT = pd.concat([risultati_MT, df_acuti_MT.loc[[acuto_str]]])
    else: print("none")

# BT
df_date_BT = df_date_final.loc['Skier-BT', :]
df_acuti_BT = df_acuti.loc['Skier-BT', :]

# Defining the df of every sample
risultati_BT = pd.DataFrame()

for i in range(6):
    g = df_date_BT.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_BT.index:
        risultati_BT = pd.concat([risultati_BT, df_acuti_BT.loc[[acuto_str]]])
    else: print("none")

# LZ
df_date_LZ = df_date_final.loc['Skier-LZ', :]
df_acuti_LZ = df_acuti.loc['Skier-LZ', :]

# Defining the df of every sample
risultati_LZ = pd.DataFrame()

for i in range(6):
    g = df_date_LZ.iloc[i]
    acuto_str = f'Acuto-{g}'
    if acuto_str in df_acuti_LZ.index:
        risultati_LZ = pd.concat([risultati_LZ, df_acuti_LZ.loc[[acuto_str]]])
    else: print("none")

# Changing the indexes of all extracted dfs
list_index = ['date1','date2', 'date3', 'date4', 'date5', 'date6']
df_list_acuto = [v for k, v in globals().items() if k.startswith("risultati")]
for dfs in df_list_acuto:
    dfs.index = list_index

# Creating the final df with acuto
df_date_e_acuto = pd.concat(df_list_acuto, axis = 1).T



# I want to do the same for cronico (I'm stupid, I know)

# DW
df_cronico_DW = df_cronico.loc['Skier-DW', :]

# Defining the df of every sample
cronico_DW = pd.DataFrame()

for i in range(6):
    g = df_date_DW.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_DW.index:
        cronico_DW = pd.concat([cronico_DW, df_cronico_DW.loc[[cronico_str]]])
    else: print("none")

# TG
df_cronico_TG = df_cronico.loc['Skier-TG', :]

# Defining the df of every sample
cronico_TG = pd.DataFrame()

for i in range(6):
    g = df_date_TG.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_TG.index:
        cronico_TG = pd.concat([cronico_TG, df_cronico_TG.loc[[cronico_str]]])
    else: print("none")

# DB
df_cronico_DB = df_cronico.loc['Skier-DB', :]

# Defining the df of every sample
cronico_DB = pd.DataFrame()

for i in range(6):
    g = df_date_DB.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_DB.index:
        cronico_DB = pd.concat([cronico_DB, df_cronico_DB.loc[[cronico_str]]])
    else: print("none")
    
# EZ
df_cronico_EZ = df_cronico.loc['Skier-EZ', :]

# Defining the df of every sample
cronico_EZ = pd.DataFrame()

for i in range(6):
    g = df_date_EZ.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_EZ.index:
        cronico_EZ = pd.concat([cronico_EZ, df_cronico_EZ.loc[[cronico_str]]])
    else: print("none")

# PB
df_cronico_PB = df_cronico.loc['Skier-PB', :]

# Defining the df of every sample
cronico_PB = pd.DataFrame()

for i in range(6):
    g = df_date_PB.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_PB.index:
        cronico_PB = pd.concat([cronico_PB, df_cronico_PB.loc[[cronico_str]]])
    else: print("none")

# DC
df_cronico_DC = df_cronico.loc['Skier-DC', :]

# Defining the df of every sample
cronico_DC = pd.DataFrame()

for i in range(6):
    g = df_date_DC.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_DC.index:
        cronico_DC = pd.concat([cronico_DC, df_cronico_DC.loc[[cronico_str]]])
    else: print("none")

# MM
df_cronico_MM = df_cronico.loc['Skier-MM', :]

# Defining the df of every sample
cronico_MM = pd.DataFrame()

for i in range(6):
    g = df_date_MM.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_MM.index:
        cronico_MM = pd.concat([cronico_MM, df_cronico_MM.loc[[cronico_str]]])
    else: print("none")

# FD
df_cronico_FD = df_cronico.loc['Skier-FD', :]

# Defining the df of every sample
cronico_FD = pd.DataFrame()

for i in range(6):
    g = df_date_FD.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_FD.index:
        cronico_FD = pd.concat([cronico_FD, df_cronico_FD.loc[[cronico_str]]])
    else: print("none")

# CC
df_cronico_CC = df_cronico.loc['Skier-CC', :]

# Defining the df of every sample
cronico_CC = pd.DataFrame()

for i in range(6):
    g = df_date_CC.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_CC.index:
        cronico_CC = pd.concat([cronico_CC, df_cronico_CC.loc[[cronico_str]]])
    else: print("none")

# IL
df_cronico_IL = df_cronico.loc['skier-IL', :]

# Defining the df of every sample
cronico_IL = pd.DataFrame()

for i in range(6):
    g = df_date_IL.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_IL.index:
        cronico_IL = pd.concat([cronico_IL, df_cronico_IL.loc[[cronico_str]]])
    else: print("none")

# DZ
df_cronico_DZ = df_cronico.loc['Skier-DZ', :]

# Defining the df of every sample
cronico_DZ = pd.DataFrame()

for i in range(6):
    g = df_date_DZ.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_DZ.index:
        cronico_DZ = pd.concat([cronico_DZ, df_cronico_DZ.loc[[cronico_str]]])
    else: print("none")

# SS
df_cronico_SS = df_cronico.loc['Skier-SS', :]

# Defining the df of every sample
cronico_SS = pd.DataFrame()

for i in range(6):
    g = df_date_SS.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_SS.index:
        cronico_SS = pd.concat([cronico_SS, df_cronico_SS.loc[[cronico_str]]])
    else: print("none")

# MT
df_cronico_MT = df_cronico.loc['Skier-MT', :]

# Defining the df of every sample
cronico_MT = pd.DataFrame()

for i in range(6):
    g = df_date_MT.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_MT.index:
        cronico_MT = pd.concat([cronico_MT, df_cronico_MT.loc[[cronico_str]]])
    else: print("none")

# BT
df_cronico_BT = df_cronico.loc['Skier-BT', :]

# Defining the df of every sample
cronico_BT = pd.DataFrame()

for i in range(6):
    g = df_date_BT.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_BT.index:
        cronico_BT = pd.concat([cronico_BT, df_cronico_BT.loc[[cronico_str]]])
    else: print("none")

# LZ
df_cronico_LZ = df_cronico.loc['Skier-LZ', :]

# Defining the df of every sample
cronico_LZ = pd.DataFrame()

for i in range(6):
    g = df_date_LZ.iloc[i]
    cronico_str = f'Cronico-{g}'
    if cronico_str in df_cronico_LZ.index:
        cronico_LZ = pd.concat([cronico_LZ, df_cronico_LZ.loc[[cronico_str]]])
    else: print("none")

# Changing the indexes of all extracted dfs
df_list_cronico = [v for k, v in globals().items() if k.startswith("cronico_")]

# There's a string in the list of dfs: I remove it
df_list_cronico = [dfss for dfss in df_list_cronico if isinstance(dfss, pd.DataFrame)]

for dfss in df_list_cronico:
    dfss.index = list_index

# Creating the final df with acuto
df_date_e_cronico = pd.concat(df_list_cronico, axis = 1).T

# Extracting from df the Wellness (aka Hooper Index) data
df_W = df.loc[:, 'W_1.1':'W_6.3']
df_RESTQ = df.iloc[:, 1:73]

#%% Some analysis on datamissing

# Calculate the missing ratio of df_W
missing = df_W.isna().sum().sum()
total = df_W.size
missing_ratio = missing / total
print(f'The missing ratio of W is {missing_ratio}')
time.sleep(2)

#%% Mean imputation to create the reference data for the evaluation
from sklearn.impute import SimpleImputer

# I want to create a reference df to evaluate the different ways of imputation
imputer = SimpleImputer(strategy='mean')
df_reference = pd.DataFrame(imputer.fit_transform(df_W), columns=df_W.columns)

# Let's remove the randomly 20% (the missing_ratio) of data from df_reference
n_total = df_reference.size  # Number of cells of reference df
n_nan = int(n_total * missing_ratio)  # Number of cells to randomly pick

nan_indices = np.random.choice(n_total, n_nan, replace=False)  # Selecting the indexes
np_flatted = df_reference.to_numpy().flatten()  # Flatted numpy array
np_flatted[nan_indices] = np.nan  # Selecting the elements and replacing them with NaN
df_Nan = pd.DataFrame(np_flatted.reshape(df_reference.shape), columns=df_W.columns)

#%% Try to use the reference and df_nan dfs to evaluate different types of imputings
from sklearn.metrics import mean_squared_error, mean_absolute_error


# KNN imputation of df_W:
from sklearn.impute import KNNImputer

KNNimputer = KNNImputer(n_neighbors=2,)
df_KNNimputed = pd.DataFrame(KNNimputer.fit_transform(df_Nan), columns = df_Nan.columns)

KNN_rmse = np.sqrt(mean_squared_error(df_KNNimputed, df_reference))
KNN_mae = mean_absolute_error(df_KNNimputed, df_reference)
print(f'The performance of KNN-imputation is RMSE = {KNN_rmse} and MAE = {KNN_mae}')
time.sleep(2)

# MICE imputation of df_W:
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

MICEimputer = IterativeImputer(max_iter=100, random_state=0)
df_MICEimputed = pd.DataFrame(MICEimputer.fit_transform(df_Nan), columns = df_Nan.columns)

MICE_rmse = np.sqrt(mean_squared_error(df_MICEimputed, df_reference))
MICE_mae = mean_absolute_error(df_MICEimputed, df_reference)
print(f'The performance of MICE-imputation is RMSE = {MICE_rmse} and MAE = {MICE_mae}')
time.sleep(2)

# Median imputator
Medianimputer = SimpleImputer(strategy='median')
df_Medianimputed = pd.DataFrame(Medianimputer.fit_transform(df_Nan), columns = df_Nan.columns)

Median_rmse = np.sqrt(mean_squared_error(df_Medianimputed, df_reference))
Median_mae = mean_absolute_error(df_Medianimputed, df_reference)
print(f'The performance of Median-imputation is RMSE = {Median_rmse} and MAE = {Median_mae}')
time.sleep(2)

# There's a lot of sampling error, so I want to bootstrap:
def bootstrap_imputer(imputer, imputer_name, num_iterations=100):
    boot_RMSEs = []
    boot_MAEs = []
    
    # Get a mask of where values are not NaN (True for valid values, False for NaN)
    valid_mask = ~df_Nan.isna()
    
    for i in range(num_iterations):
        # Initialize a DataFrame to hold the sampled data
        sampled_data = pd.DataFrame(index=df_Nan.index, columns=df_Nan.columns)
        sampled_reference = pd.DataFrame(index=df_reference.index, columns=df_reference.columns)
        
        # Loop through each column
        for col in df_Nan.columns:
            # Get the valid (non-NaN) values in the current column
            valid_values = df_Nan[col][valid_mask[col]]
            valid_reference = df_reference[col][valid_mask[col]]  # Corresponding valid reference values
            
            # Initialize the column with the original data
            sampled_data[col] = df_Nan[col]
            sampled_reference[col] = df_reference[col]
            
            # If there are valid values, sample with replacement
            if not valid_values.empty:
                sampled_values = np.random.choice(valid_values, size=valid_mask[col].sum(), replace=True)
                sampled_ref_values = np.random.choice(valid_reference, size=valid_mask[col].sum(), replace=True)
                
                # Only replace the valid (non-NaN) values with the sampled values
                sampled_data.loc[valid_mask[col], col] = sampled_values
                sampled_reference.loc[valid_mask[col], col] = sampled_ref_values
                
            else:
                # If no valid values, keep the column as NaN
                sampled_data[col] = np.nan
                sampled_reference[col] = np.nan
                
        # Perform imputation on the sampled data
        boot_imputed = imputer.fit_transform(sampled_data)  
        
        # Calculate RMSE and MAE comparing imputed data to the reference (true) data
        boot_RMSE = np.sqrt(mean_squared_error(sampled_reference, boot_imputed))
        boot_MAE = mean_absolute_error(sampled_reference, boot_imputed)
        
        # Store the results
        boot_RMSEs.append(boot_RMSE)
        boot_MAEs.append(boot_MAE)
        
    # Calculate 95% Confidence Intervals for RMSE and MAE
    CI_RMSE = np.percentile(boot_RMSEs, [2.5, 97.5])
    CI_MAE = np.percentile(boot_MAEs, [2.5, 97.5])
    # Restituisci gli intervalli di confidenza
    
    # Print the results with formatted confidence intervals
    results = {
        f'The confidence interval of {imputer_name} for RMSE': CI_RMSE,
        f'The confidence interval of {imputer_name} for MAE': CI_MAE
    }
    
    for key, value in results.items():
        print(f'{key}: {value}')
    # Return the results to be used later on
    return {
        'CI_RMSE': CI_RMSE,
        'CI_MAE': CI_MAE
    }

# Def to selct the best imputation strategy
def precision_ic(ic):
    return ic[1]-ic[0]

# Bootstrapping with KNN
ic_KNN = bootstrap_imputer(KNNimputer, 'KNNImputer')
time.sleep(2)

length_KNN_RMSE = precision_ic(ic_KNN['CI_RMSE'])  # Lenght of the ic
length_KNN_MAE = precision_ic(ic_KNN['CI_MAE'])  # Lenght of the ic

# Bootstrapping with Meadian
ic_Median = bootstrap_imputer(Medianimputer, 'Medianimputer')
time.sleep(2)

length_Median_RMSE = precision_ic(ic_Median['CI_RMSE'])  # Lenght of the ic
length_Median_MAE = precision_ic(ic_Median['CI_MAE'])  # Lenght of the ic

# Bootstrapping with MICE
#ic_MICE = bootstrap_imputer(MICEimputer, 'MICEimputer')
#time.sleep(2)

#length_MICE_RMSE = precision_ic(ic_MICE['CI_RMSE'])  # Lenght of the ic
#length_MICE_MAE = precision_ic(ic_MICE['CI_MAE'])  # Lenght of the ic

# Comparing the leghts of the ICs
methods_ICs = {
    'KNN': {
        'Lenght CI RMSE': length_KNN_RMSE,
        'Lenght CI MAE': length_KNN_MAE
        
        },
    'Median': {
        'Lenght CI RMSE': length_Median_RMSE,
        'Lenght CI MAE': length_Median_MAE
        },
    'MICE': {
#        'Lenght CI RMSE': length_MICE_RMSE,
#        'Lenght CI MAE': length_MICE_MAE
        }
    }
def best_method(compare_ic):
    valid_methods = {k: v for k, v in compare_ic.items() if 'Lenght CI RMSE' in v and 'Lenght CI MAE' in v}
    bestRMSE = min(valid_methods, key = lambda x: methods_ICs[x]['Lenght CI RMSE'])
    bestMAE = min(valid_methods, key = lambda x: methods_ICs[x]['Lenght CI MAE'])
    return bestRMSE, bestMAE

bestRMSE, bestMAE = best_method(methods_ICs)
print(f'The best method for RMSE is {bestRMSE}')
time.sleep(2)
print(f'The best method for MAE is {bestMAE}')
time.sleep(2)
if bestRMSE == bestMAE:
    print(f"Let's take {bestRMSE} for the imputation")
else:
    print("Think about imputation method")

#%% I have to evaluate the imputation also for RESTQ
# I'll create a df_RESTQ_reference and the df_RESTQ_nan as dfs to evaluate imputing

time.sleep(2)
print('Now considering RESTQ values...')
time.sleep(2)
# Create the reference df to evaluate the imputations
df_RESTQ_reference = pd.DataFrame(imputer.fit_transform(df_RESTQ),
                                  columns=df_RESTQ.columns)

# Missing index of the df_RESTQ
missing_ratio_RESTQ = round(df_RESTQ.isna().sum().sum() / df_RESTQ.size, 3)
print(f'The missing ratio of RESTQ is {missing_ratio_RESTQ}')
time.sleep(2)

n_nan_RESTQ = df_RESTQ.isna().sum().sum()

sample_time_nan = {'Sample':[], 'Time':[]} # in this dictionaty I'll put all the samples' indexes that will be nan

for i in range(4):
    samples_nan = random.randint(0, 14)
    sample_time_nan['Sample'].append(samples_nan)

for i in range(4):
    times_nan = random.randint(0, 6)
    sample_time_nan['Time'].append(times_nan*12)

# These are the tuples that indentiy the start of the missing RESTQ values
paired_samples_times = list(zip(sample_time_nan['Sample'], sample_time_nan['Time']))

df_RESTQ_nan = df_RESTQ_reference.copy()  # df_RESTQ_nan is the df to use to evaluate imputations

for i in range(4):  # I want to select the cells and replace them with Nan
   sample_index = sample_time_nan['Sample'][i]
   time_index = sample_time_nan['Time'][i]
   # Set the cell (sample_index, time_index) and the next 11 cells to NaN (they have to be packed)
   df_RESTQ_nan.iloc[sample_index, time_index:time_index + 12] = np.nan

#%% Let's try different imputation strategies on RESTQ

# Define a boostrap function for RESTQ
def bootstrap_imputer_RESTQ(imputer, imputer_name, num_iterations=100):
    boot_RMSEs = []
    boot_MAEs = []
    
    # Get a mask of where values are not NaN (True for valid values, False for NaN)
    valid_mask = ~df_RESTQ_nan.isna()
    
    for i in range(num_iterations):
        # Initialize a DataFrame to hold the sampled data
        sampled_data = pd.DataFrame(index=df_RESTQ_nan.index, columns=df_RESTQ_nan.columns)
        sampled_reference = pd.DataFrame(index=df_RESTQ_reference.index, columns=df_RESTQ_reference.columns)
        
        # Loop through each column
        for col in df_RESTQ_nan.columns:
            # Get the valid (non-NaN) values in the current column
            valid_values = df_RESTQ_nan[col][valid_mask[col]]
            valid_reference = df_RESTQ_reference[col][valid_mask[col]]  # Corresponding valid reference values
            
            # Initialize the column with the original data
            sampled_data[col] = df_RESTQ_nan[col]
            sampled_reference[col] = df_RESTQ_reference[col]
            
            # If there are valid values, sample with replacement
            if not valid_values.empty:
                sampled_values = np.random.choice(valid_values, size=valid_mask[col].sum(), replace=True)
                sampled_ref_values = np.random.choice(valid_reference, size=valid_mask[col].sum(), replace=True)
                
                # Only replace the valid (non-NaN) values with the sampled values
                sampled_data.loc[valid_mask[col], col] = sampled_values
                sampled_reference.loc[valid_mask[col], col] = sampled_ref_values
                
            else:
                # If no valid values, keep the column as NaN
                sampled_data[col] = np.nan
                sampled_reference[col] = np.nan
                
        # Perform imputation on the sampled data
        boot_imputed = imputer.fit_transform(sampled_data)  
        
        # Calculate RMSE and MAE comparing imputed data to the reference (true) data
        boot_RMSE = np.sqrt(mean_squared_error(sampled_reference, boot_imputed))
        boot_MAE = mean_absolute_error(sampled_reference, boot_imputed)
        
        # Store the results
        boot_RMSEs.append(boot_RMSE)
        boot_MAEs.append(boot_MAE)
        
    # Calculate 95% Confidence Intervals for RMSE and MAE
    CI_RMSE = np.percentile(boot_RMSEs, [2.5, 97.5])
    CI_MAE = np.percentile(boot_MAEs, [2.5, 97.5])
    # Restituisci gli intervalli di confidenza
    
    # Print the results with formatted confidence intervals
    results = {
        f'The confidence interval of {imputer_name} for RMSE': CI_RMSE,
        f'The confidence interval of {imputer_name} for MAE': CI_MAE
    }
    
    for key, value in results.items():
        print(f'{key}: {value}')
    # Return the results to be used later on
    return {
        'CI_RMSE': CI_RMSE,
        'CI_MAE': CI_MAE
    }


# Bootstrapping with KNN
ic_RESTQ_KNN = bootstrap_imputer_RESTQ(KNNimputer, 'KNNImputer')
time.sleep(2)

length_KNN_RESTQ_RMSE = precision_ic(ic_RESTQ_KNN['CI_RMSE'])  # Lenght of the ic
length_KNN_RESTQ_MAE = precision_ic(ic_RESTQ_KNN['CI_MAE'])  # Lenght of the ic

# Bootstrapping with Meadian
ic_RESTQ_Median = bootstrap_imputer_RESTQ(Medianimputer, 'Medianimputer')
time.sleep(2)

length_Median_RESTQ_RMSE = precision_ic(ic_RESTQ_Median['CI_RMSE'])  # Lenght of the ic
length_Median_RESTQ_MAE = precision_ic(ic_RESTQ_Median['CI_MAE'])  # Lenght of the ic

# Bootstrapping with MICE
#ic_RESTQ_MICE = bootstrap_imputer_RESTQ(MICEimputer, 'MICEimputer')
#time.sleep(2)

#length_MICE_RESTQ_RMSE = precision_ic(ic_RESTQ_MICE['CI_RMSE'])  # Lenght of the ic
#length_MICE_RESTQ_MAE = precision_ic(ic_RESTQ_MICE['CI_MAE'])  # Lenght of the ic

# Comparing the leghts of the ICs
methods_RESTQ_ICs = {
    'KNN': {
        'Lenght CI RMSE': length_KNN_RESTQ_RMSE,
        'Lenght CI MAE': length_KNN_RESTQ_MAE
        
        },
    'Median': {
        'Lenght CI RMSE': length_Median_RESTQ_RMSE,
        'Lenght CI MAE': length_Median_RESTQ_MAE
        },
    'MICE': {
#        'Lenght CI RMSE': length_MICE_RESTQ_RMSE,
#        'Lenght CI MAE': length_MICE_RESTQ_MAE
        }
    }

bestRMSE_RESTQ, bestMAE_RESTQ = best_method(methods_RESTQ_ICs)
print(f'The best method for RMSE is {bestRMSE_RESTQ}')
time.sleep(2)
print(f'The best method for MAE is {bestMAE_RESTQ}')
time.sleep(2)
if bestRMSE_RESTQ == bestMAE_RESTQ:
    print(f"Let's take {bestRMSE_RESTQ} for the imputation")
else:
    print("Think about imputation method")

#%% I've chosen the imputation methods, let's impute the official dfs
df_W_imputed = pd.DataFrame(KNNimputer.fit_transform(df_W),
                            index=df_W.index, columns=df_W.columns)  # This is the imputed df of W

df_RESTQ_imputed = pd.DataFrame(KNNimputer.fit_transform(df_RESTQ),
                                index=df_RESTQ.index, columns=df_RESTQ.columns)  # This is the imputed df of RESTQ

# Let's do the factor values for W and RESTQ (aka sum of the days)
df_RESTQ_factors = []
for j in range(df_RESTQ_imputed.shape[0]):  # for every row
    row_factors = []  # List of the row's factors
    for i in range(6):
        column_to_finish = (i + 1) * 12 # Last column (of every pack)
        column_to_start = column_to_finish - 12  # First column (of every pack)
        
        factor = df_RESTQ_imputed.iloc[j, column_to_start:column_to_finish].sum()
        row_factors.append(factor)
    df_RESTQ_factors.append(row_factors)
df_RESTQ_factors2 = pd.DataFrame(df_RESTQ_factors,
                                index=df_RESTQ.index, columns=range(1,7))

df_W_factors = []
for j in range(df_W_imputed.shape[0]):  # for every row
    row_factors = []  # List of the row's factors
    for i in range(6):
        column_to_finish = (i + 1) * 3  # Last column (of every pack)
        column_to_start = column_to_finish - 3 # First column (of every pack)

        factor = df_W_imputed.iloc[j, column_to_start:column_to_finish].mean()
        row_factors.append(factor)
    df_W_factors.append(row_factors)
df_W_factors2 = pd.DataFrame(df_W_factors,
                                index=df_W.index, columns=range(1,7))


# Acuti mean and SD
df_acuti_mean = df_acuti.mean()
df_acuti_std = df_acuti.std()

df_acuti_mean = pd.DataFrame(df_acuti_mean)
df_acuti_std = pd.DataFrame(df_acuti_std)

# Number of 0s in TL
df2_TL = df2.filter(like='TL-').drop(
    columns=['TL- 21/09','TL- 22/09','TL- 23/09','TL- 24/09','TL- 25/09','TL- 26/09','TL- 27/09','TL- 01/10','TL- 02/10','TL- 03/10','TL- 04/10','TL- 05/10','TL- 06/10','TL- 11/10','TL- 14/10','TL- 17/10','TL- 18/10','TL- 23/10','TL- 24/10','TL- 25/10','TL- 26/10','TL- 27/10','TL- 28/10'])
columns_name_TL = [df_acuti.columns]
zero_frac = pd.DataFrame(((df2_TL == 0).sum())/df2_TL.shape[0])
zero_frac.index = columns_name_TL

#%% Normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# I want to normalize all the data
df_RESTQ_factors = pd.DataFrame(scaler.fit_transform(df_RESTQ_factors2),
                                index = df_RESTQ_factors2.index, columns = df_RESTQ_factors2.columns)
df_W_factors = pd.DataFrame(scaler.fit_transform(df_W_factors2),
                            index = df_W_factors2.index, columns = df_W_factors2.columns)
df_acuti_mean = pd.DataFrame(scaler.fit_transform(df_acuti_mean),
                             index = df_acuti_mean.index)
df_date_e_acuto = pd.DataFrame(scaler.fit_transform(df_date_e_acuto), 
                               index = df_date_e_acuto.index, columns = df_date_e_acuto.columns)
df_acuti_std = pd.DataFrame(scaler.fit_transform(df_acuti_std),
                            index=df_acuti_std.index, columns=['Standard_Deviation'])

#%% Create a plot to see the trends
time.sleep(2)
print("..see the trend's plot")

# This is a plot for W_factors
sns.set(style="white")
for idx, row in df_W_factors2.iterrows():
    plt.plot(df_W_factors2.columns, row, marker='o')

plt.xticks(range(6), ['1_W','2_W','3_W','4_W','5_W','6_W' ])
plt.xlabel('Time')
plt.ylabel('W_mean')
plt.title('Graph W in samples')

# plt.legend(title='Samples', loc='upper right')
plt.show()

# This is a plot for RESTQ_factors
sns.set(style="white")
for idx, row in df_RESTQ_factors2.iterrows():
    plt.plot(df_RESTQ_factors2.columns, row, marker='o')  # If you want add 'label=idx'

plt.xticks(range(6), ['1_RESTQ','2_RESTQ','3_RESTQ','4_RESTQ','5_RESTQ','6_RESTQ'])
plt.xlabel('Time')
plt.ylabel('RESTQ_mean')
plt.title('Graph RESTQ in samples')

# plt.legend(title='Samples', loc='lower right')
plt.show()

# Create a plot for Acuto data of df_acuti
sns.set(style="white")
for idx, row in df_acuti.iterrows():
    plt.plot(df_acuti.columns, row, marker='o')

plt.xlabel('days')
plt.ylabel('Acuti')
plt.title('Graph Acuti in samples')

# plt.legend(title='Samples', loc='upper right')
plt.show()

# Create a plot for Acuto data of df_acuti
sns.set(style="white")
fig, ax1 = plt.subplots()
ax1.plot(df_acuti_mean.index, df_acuti_mean, label='Media', color='blue')

df_acuti_mean = df_acuti_mean.squeeze()
df_acuti_std = df_acuti_std.squeeze()
df_acuti_std.squeeze()
# Aggiungi band of DS
ax1.fill_between(df_acuti_mean.index,
                 df_acuti_mean - df_acuti_std, 
                 df_acuti_mean + df_acuti_std, 
                 color='blue', alpha=0.2, label='±1 SD')

column_indices = [0, 5, 9, 14, 18, 20]  # Columns of the values to plot on x
list_lable = ['1_Acuto','2_Acuto','3_Acuto','4_Acuto','5_Acuto','6_Acuto']
plt.xticks(column_indices, list_lable, rotation=45)

plt.title("Acute's mean with ±1 SD")
plt.xlabel('Time')
plt.ylabel('Acute')

plt.legend()
plt.grid()

ax2 = ax1.twinx()  # The second axis
ax2.plot(df_acuti_mean.index, zero_frac, color='red', alpha=0.3)
ax2.set_ylim(0, 1)
ax2.set_ylabel('Zeros', color='red')
ax2.set_xlabel('Second Time Axis')
ax2.set_ylabel('Zeros', color='red')

fig.tight_layout()
plt.show()

# Create a scatterplor between Zeroes and SD
sns.set(style="white")
zero_frac_serie = zero_frac.squeeze()
zero_frac_serie.index = df_acuti_std.index
data = pd.DataFrame({'SD': df_acuti_std.values.ravel(), 'Zeroes': zero_frac_serie}, index = df_acuti_std.index)  # Create a df on data

plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='SD', y='Zeroes', color='blue', scatter_kws={'alpha':0.5}, line_kws={'color':'red'})

plt.title('Scatter Plot between Zeros and SD')
plt.xlabel('Standard Deviation (SD)')
plt.ylabel('Zeroes')

plt.grid()
plt.show()
#%% Statistics on the data

##### Heatmap of correlations #####

# Set the same columns and indxes
df_RESTQ_factors.columns = df_date_e_acuto.columns
df_RESTQ_factors.index = df_date_e_acuto.index
df_W_factors.columns = df_date_e_acuto.columns
df_W_factors.index = df_date_e_acuto.index

# Computing the single correlations
correlation_acuto_RESTQ_1 = df_RESTQ_factors.corrwith(df_date_e_acuto, axis=1)
correlation_acuto_RESTQ_1 = pd.DataFrame(correlation_acuto_RESTQ_1, columns = ['r acuto-RESTQ'])

correlation_acuto_W_1 = df_W_factors.corrwith(df_date_e_acuto, axis=1)
correlation_acuto_W_1 = pd.DataFrame(correlation_acuto_W_1, columns = ['r acuto-W'])

correlation_W_RESTQ_1 = df_RESTQ_factors.corrwith(df_W_factors, axis=1)
correlation_W_RESTQ_1 = pd.DataFrame(correlation_W_RESTQ_1, columns = ['r W-RESTQ'])

# Doing the mean through the samples
correlation_acuto_RESTQ_m = correlation_acuto_RESTQ_1.mean().item()
correlation_acuto_W_m = correlation_acuto_W_1.mean().item()
correlation_W_RESTQ_m = correlation_W_RESTQ_1.mean().item()

# Build the df of the corrlation matrix
df_correlation_total = pd.DataFrame(
    {
    'Acuto': [1, 0, 0],
    'RESTQ': [correlation_acuto_RESTQ_m, 1, 0],
    'W': [correlation_acuto_W_m, correlation_W_RESTQ_m, 1]
}, index=['Acuto', 'RESTQ', 'W'])


# Create the heatmap with covered 0
sns.set(style="white")
plt.figure(figsize=(8, 6))

mask_zero = df_correlation_total.T == 0  # The mask to cover 0
sns.heatmap(df_correlation_total.T, annot=True, mask=mask_zero, cmap='coolwarm', cbar=True)

plt.title('Correlation Matrix')
plt.show()


##### Signal-to-Noise Ratio #####
df_W_factor_mean = []
df_W_factor_ST = []
for col in df_W_factors.columns:
    mean_col = df_W_factors[col].mean()
    df_W_factor_mean.append(mean_col)
    
    ST_col = df_W_factors[col].std()
    df_W_factor_ST.append(ST_col)

df_RESTQ_factor_mean = []
df_RESTQ_factor_ST = []

for col in df_RESTQ_factors.columns:
    mean_col = df_RESTQ_factors[col].mean()
    df_RESTQ_factor_mean.append(mean_col)
    
    ST_col = df_RESTQ_factors[col].std()
    df_RESTQ_factor_ST.append(ST_col)

time.sleep(2)
acuti_weekly = pd.DataFrame(df_acuti_mean.iloc[[0, 5, 9, 13, 17, 20]]) 

df_RESTQ_factor_mean = pd.DataFrame(df_RESTQ_factor_mean)
variance_RESTQ = np.var(df_RESTQ_factor_mean, axis = 0)

df_W_factor_mean = pd.DataFrame(df_W_factor_mean)
variance_W = np.var(df_W_factor_mean, axis = 0)

smse_AC_RESTQ = mean_squared_error(df_RESTQ_factor_mean, acuti_weekly) / variance_RESTQ
smse_AC_W = mean_squared_error(df_W_factor_mean, acuti_weekly) / variance_W
smse_AC_RESTQ_scalar = smse_AC_RESTQ.item()  # Utilizza .item() se è una Serie di un solo elemento
smse_AC_W_scalar = smse_AC_W.item()
if smse_AC_RESTQ_scalar > smse_AC_W_scalar:
    print('W is better than RESTQ in detecting Training Load fluctuations')
else:
    print('W is worse than RESTQ in detecting Training Load fluctuations')
time.sleep(2)
print(f'MSE of RESTQ is {smse_AC_RESTQ_scalar}')
print(f'MSE of W is {smse_AC_W_scalar}')
time.sleep(2)

#%% ML classifiers

#### SETTING THE DF FOR CLUSTER ANALYSIS ####
from sklearn.cluster import AffinityPropagation

def col_name(variable):  # Function to easily create the list of the columns
    variable_list = []
    for i in range(1,7):
        name = f'date_{i}_{variable}'
        variable_list.append(name)
    return variable_list

RESTQ_list = col_name('RESTQ')
W_list = col_name('W')

df = df.reset_index(drop=True)

df_W_factors = df_W_factors.reset_index(drop=True)
df_W_factors.columns = W_list

df_RESTQ_factors = df_RESTQ_factors.reset_index(drop=True)
df_RESTQ_factors.columns = RESTQ_list

df_total = pd.concat([df.iloc[:,0], df_W_factors, df_RESTQ_factors], axis=1)
df_total.index = df_W.index  # This is the df to use

df_total = pd.get_dummies(df_total, columns=['Genere'], drop_first=True)  # Convert sex into dummy

#### TRYING AFFINITY PROPAGATION MODEL ####
AfPrModel = AffinityPropagation(max_iter=250)

Clusters = AfPrModel.fit(df_total)  # Fitting the model into the df
clusters_prediction = pd.DataFrame(AfPrModel.predict(df_total), index = df_W.index, columns = ['cluster'])
cluster_centers_indices = AfPrModel.cluster_centers_indices_
n_clusters = len(cluster_centers_indices)

print(n_clusters)
print(clusters_prediction.columns)

cluster_members = {
    '0' : [],
    '1' : [],
    '2' : [],
    '3' : [],
    '4' : []}

for idx, row in clusters_prediction.iterrows():
    cluster_label = row['cluster']
    cluster_members[str(cluster_label)].append(idx)

print(f'The clusters of Affinity propagation are: {cluster_members}')
print("\n They don't make sense")  # Maybe due to little sample size and lot of noise

#### TRY PCA ####
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  # Standardize
scaled_data = scaler.fit_transform(df_total)

new_name = []  # This is the list of the samples to use to avoid named (for privacy reasons)
for i in range(1,16):
    name = f'Skier-{i}'
    new_name.append(name)
pca = PCA(n_components=2)  # Number of components to keep
pca_result = pca.fit_transform(scaled_data)


loadings = pca.components_.T
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=df_total.columns)

# Visualize the Loadings for variables
sns.set(style="white")
plt.figure(figsize=(5, 6))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings')
plt.xlabel('Principal Components')
plt.ylabel('Variables')
plt.show()
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index = df_W.index)


# I want  to do a PCA with .T to see how it devides the sample
pca2 = PCA(n_components=2)
pca2_result = pca2.fit_transform(scaled_data.T)

loadings_T = pca2.components_.T
loadings_df_T = pd.DataFrame(loadings_T, columns=['PC1', 'PC2'], index=new_name)

# Visualize the loadings for sample
sns.set(style="white")
plt.figure(figsize=(5, 6))
sns.heatmap(loadings_df_T, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings for samples')
plt.xlabel('Principal Components')
plt.ylabel('Samples')
plt.show()

# I want  to do a PCA with .T to see how it devides the sample without Sex
pca3 = PCA(n_components=2)
scaled_data= pd.DataFrame(scaled_data)
scaled_data_2 = scaled_data.iloc[:, 0:12]
pca3_result = pca3.fit_transform(scaled_data_2.T)

loadings_T2 = pca3.components_.T
loadings_df_T2 = pd.DataFrame(loadings_T2, columns=['PC1', 'PC2'], index=new_name)

# Visualize the loadings for sample
sns.set(style="white")
plt.figure(figsize=(5, 6))
sns.heatmap(loadings_df_T2, annot=True, cmap='coolwarm', center=0)
plt.title('PCA Loadings for samples')
plt.xlabel('Principal Components')
plt.ylabel('Samples')
plt.show()

#### TRY SUPERVISED CLUSTER ####

# Try supervised cluster and see if the importances make sense
from sklearn import tree
clf = tree.DecisionTreeClassifier()

groups = {
    'M1': ['Skier-DW', 'Skier-TM', 'Skier-DB','Skier-EZ'],
    'M2': ['Skier-PB','Skier-DC', 'Skier-MM', 'Skier-FD', 'Skier-CC', 'Skier-IL', 'Skier-DZ'],
    'F': ['Skier-SS', 'Skier-MT', 'Skier-BT', 'Skier-LZ']}
groups_sample = [(skier, group) for group, skiers in groups.items() for skier in skiers]
target_group = pd.DataFrame(groups_sample, columns = ['Name', 'Group']).set_index('Name')

clf = clf.fit(df_total,target_group)  # Fitting the model

features = df_total.columns
importance = clf.feature_importances_  # Impurity-based importance
PVI = pd.DataFrame({'Feature':features, 'Importance': importance})
importance_df = PVI.sort_values(by='Importance', ascending=False)
print(importance_df)
