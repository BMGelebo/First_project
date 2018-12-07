
""" 
This code is to learn the usage of Github 

"""
# Import basic necessary libraries 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import scipy as sc


import warnings
warnings.filterwarnings("ignore")


# Read datasets from the directory where you saved 
data = pd.read_csv('customers.csv', index_col= 0)


# check missing values
data.isnull().sum() # The column CRG is with missing values of 5537

''' ------------Missing value handling------------------  '''
# Drop clients who have not given CRG for 32 months 
def fill_missing_values(df):
    column = ['col']  # Column with missing values
    nr_crg =df.groupby(['id'])['col'].apply( # Number of months a cleint have
        lambda x: x.isnull().sum())
    ind_less = nr_crg[nr_crg<31].index.values  # drop itiems with this index from orginal data
    crg_32_dropped = data.loc[data['id'].isin(ind_less)] 
    # Fill the missing value with maximum grade a client assigned
    for column in crg_32_dropped[column]:
        crg_32_dropped[column]=crg_32_dropped.groupby(['id'])[column].ffill().bfill()
        if crg_32_dropped[column].isnull().sum()>0:
            crg_32_dropped[column]= crg_32_dropped[column].fillna(
                    crg_32_dropped[column].max())
    return crg_32_dropped

# PCA on the datasets, choose one of the datasets based on information they contain 
def explained_variance(df, n_components):
    features = list(set(df.columns) - set(['v1', 'id', 'year']))
    X =df[features] 
    X_scaled = pd.DataFrame(scale(X.astype(float)),
                            columns = X.columns)
    pca = PCA(n_components)
    num_pca = pca.fit_transform(X_scaled)
    X_scaled_pca = pd.DataFrame(pca.transform(X_scaled),
                            columns = ['PC_' + str(i) for i in range(n_components)])
    print('The explained cumulative variance is',
    np.round(pca.explained_variance_ratio_.sum()*100,2),'%')

# Call the function on each datasets
explained_variance(data,5),

## Function aggregate data either at clinet or yearmonth level

def create_features(df, level):
    """The function agregate data frame (df) at different level of granularity
    
    Args:
    df:    Dataframe to be aggregated
    level: Group at which each column of dataframe aggregated ('client_nr' or 'yearmonth')
    
    Returns: 
    A dataframe with rows of unique number of level and columns of a multiple of number 
    of features and number of aggregationfunctions.Functions such as sum, minimum, ...
    are applied to each column (feature)"""
    
    df_grouped = df.groupby(
    level).agg({
        'yearmonth': ['count'],
        'client_nr':['count'],
    'total_nr_trx':[min,'mean', max, sum],
    'nr_debit_trx':[min,'mean', max, sum],
    'volume_debit_trx':[min,'median','mean', max,'std','sem', sum],
    'nr_credit_trx':[min,'mean', max, sum],
    'volume_credit_trx':[min,'median','mean', max,'std','sem','sum'],
    'min_balance':[min,'median','mean', max,'std','sem', sum],
    'max_balance':[min,'median','mean', max,'std','sem', sum],
    #'CRG':[min,'median', max],
    'credit_application':[max, sum],
    'nr_credit_applications':[min, max, sum]
    })
    
    df_grouped.columns = ["_".join(x) for x in df_grouped.columns.ravel()]
    
    # For level = 'id' calculate autoccoralation coffecient at lag 1 and 2
    if (level == 'client_nr'):
        clmn = ['total_nr_trx','nr_debit_trx','volume_debit_trx', 'nr_credit_trx', 
                'volume_credit_trx', 'min_balance', 'max_balance']
        for col in clmn:
            df_grouped[str(col) + '_lag1corr'] = df.groupby(
                level)[col].apply(lambda x: x.autocorr(1)) # autoccoralation at lag 1
            df_grouped[str(col) + '_lag2corr'] = df.groupby(
                level)[col].apply(lambda x: x.autocorr(2)) # autoccoralation at lag 2
    return df_grouped

# Aggregate the datasets at client and monthly level
aggr_data_id = create_features(data, 'id')
aggr_data_year = create_features(data, 'year')

