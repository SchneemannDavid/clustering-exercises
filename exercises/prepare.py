import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from env import user, password, host

# ----------------------------------------------------------------------------------# 

# Clustering prepare file (prepare.py)
'''
file description
'''


# Potential function for removing properties other than single unit properties
def trim_bad_data_zillow(df):
    # If it's not single unit, it's not a single family home.
    df = df[~(df.unitcnt > 1)]
    # If the lot size is smaller than the finished square feet, it's probably bad data or not a single family home
    df = df[~(df.lotsizesquarefeet < df.calculatedfinishedsquarefeet)]
    # If the finished square feet is less than 500 it is likeley an apartment, or bad data
    df = df[~(df.calculatedfinishedsquarefeet < 500)]
    # If there are no bedrooms, likely a loft or bad data
    df = df[~(df.bedroomcnt < 1)]
    # Drop duplicate parcels
    df = df.drop_duplicates(subset='parcelid')
    return df


#Dropping columns with an inordinate number of nulls (rendering variable essentially useless)
def drop_bad_columns(df):
    df = df.drop(columns = [
        'typeconstructiontypeid',
        'heatingorsystemtypeid',
        'buildingclasstypeid',
        'architecturalstyletypeid',
        'airconditioningtypeid',
        'propertylandusetypeid',
        'basementsqft',
        'decktypeid',
        'finishedfloor1squarefeet',
        'finishedsquarefeet13',
        'finishedsquarefeet15',
        'finishedsquarefeet50',
        'finishedsquarefeet6',
        'fireplacecnt',
        'hashottuborspa',
        'poolsizesum',
        'pooltypeid10',
        'pooltypeid2',
        'pooltypeid7',
        'storytypeid',
        'yardbuildingsqft17',
        'yardbuildingsqft26',
        'numberofstories',
        'fireplaceflag',
        'taxdelinquencyflag',
        'taxdelinquencyyear',
        'architecturalstyledesc',
        'buildingclassdesc',
        'typeconstructiondesc',
        'buildingqualitytypeid',
        'propertyzoningdesc',
        'rawcensustractandblock',
        'regionidneighborhood',
        'threequarterbathnbr',
        'airconditioningdesc',
        'heatingorsystemdesc',
        'threequarterbathnbr'
        ], 
        axis=1)
    return df


def drop_nulls(df):
    # Change all remaining null/nan values to 0 or the variable mean, depending on best use case
    df['poolcnt'] = df['poolcnt'].fillna(0)
    df['garagecarcnt'] = df['garagecarcnt'].fillna(0)
    df['garagetotalsqft'] = df['garagetotalsqft'].fillna(0)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].fillna(value=df['lotsizesquarefeet'].mean())
    df['regionidcity'] = df['regionidcity'].fillna(0)
    df['unitcnt'] = df['unitcnt'].fillna(0)
    return df


def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df

def optimize_types(df):
    # Convert some columns to integers for optimization
    # fips, yearbuilt, and bedrooms, taxvaluedollarcnt, and calculatedfinishedsquarefeet can be integers
    df["fips"] = df["fips"].astype(int)
    df["yearbuilt"] = df["yearbuilt"].astype(int)
    df["bedroomcnt"] = df["bedroomcnt"].astype(int)    
    df["taxvaluedollarcnt"] = df["taxvaluedollarcnt"].astype(int)
    df["calculatedfinishedsquarefeet"] = df["calculatedfinishedsquarefeet"].astype(int)
    return df


def handle_outliers(df):
    """Manually handle outliers that do not represent properties likely for 99% of buyers and zillow visitors"""
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 1_500_000]

    return df

def clean_variables(df):
    # Drop 'taxamount' column (variable is inconsistent based on time and location of collected value, could lead to poor analysis)
    # Rename columns and 'fips' values to reflect actual location (to solidify column as categorical variable)
    df = df.drop(columns = 'taxamount')
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'sq_ft', 
                              'taxvaluedollarcnt':'home_value', 
                              'yearbuilt':'year_built', 
                              'fips':'location',
                              'fullbathcnt':'full_bathrooms',
                              'garagecarcnt':'garage_spaces',
                              'lotsizesquarefeet':'lot_sq_ft'
                             })
    df.location = df.location.replace(to_replace={6037:'LA County', 6059:'Orange County', 6111:'Ventura County'})

    return df 

def feature_engineering(df):
    # Bin `year_built` by decade
    df["decade_built"] = pd.cut(x=df["year_built"], bins=[1800, 1899, 1909, 1919, 1929, 1939, 1949, 1959, 1969, 1979, 1989, 1999, 2009], labels=['1800s', '1900s', '10s', '20s', '30s', '40s', '50s', '60s', '70s', '80s', '90s', '2000s'])
    # Convert categorical variable to numeric var
    df['county_encoded'] = df.location.map({'LA County': 0, 'Orange County': 1, 'Ventura County': 2})

    df = df.dropna()

    return df

# Split for Exploration

## 
# Train, Validate, Test Split Function: for exploration
def zillow_split_explore(df):
    '''
    This function performs split on telco data, stratifying on churn.
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3,
                                   random_state=123)
    return train, validate, test

### ------------------------------------------------------------------------

# Split for Modeling: X & Y dfs
def zillow_split_model(df):
    '''
    This function performs split on telco data, stratifying on churn.
    Returns both X and y train, validate, and test dfs
    '''
    
    train_validate, test = train_test_split(df, test_size=.2,
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3,
                                   random_state=123)

    # Splitting train, validate, and test dfs on x and y
    x_train = train.drop(columns=['home_value'])
    x_validate = validate.drop(columns=['home_value'])
    x_test = test.drop(columns=['home_value'])

    y_train = train['home_value']
    y_validate = validate['home_value']
    y_test = test['home_value']
    
    return x_train, y_train, x_validate, y_validate, x_test, y_test


def prep_zillow(df):
    """
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    clean variables via dropping columns and renaming features
    includes feature engineering 
    returns a clean dataframe
    Splits df into train, validate, test, and associated dfs on x and y 
    """
    df = trim_bad_data_zillow(df)

    df = drop_bad_columns(df)

    df = drop_nulls(df)

    df = handle_nulls(df)

    df = optimize_types(df)

    df = handle_outliers(df)

    df = clean_variables(df)

    df = feature_engineering(df)

    train, validate, test = zillow_split_explore(df)

    x_train, y_train, x_validate, y_validate, x_test, y_test = zillow_split_model(df)

    # df.to_csv("zillow.csv", index=False)

    return df, train, validate, test, x_train, y_train, x_validate, y_validate, x_test, y_test


# ----------------------------------------------------------- # 

############## Regression Project acquire.py for reference (now: only what remains after transfer to new file above!)

def handle_nulls(df):    
    # We keep 99.41% of the data after dropping nulls
    # round(df.dropna().shape[0] / df.shape[0], 4) returned .9941
    df = df.dropna()
    return df

