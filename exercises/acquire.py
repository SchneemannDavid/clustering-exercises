import pandas as pd
import numpy as np
import os
from env import user, password, host

def get_db_url(database):
    return f'mysql+pymysql://{user}:{password}@{host}/{database}'

"""
USAGE: 
Use `from wrangle import wrangle_zillow` at the top of your notebook.
This 
"""
def get_zillow_data():
    """Seeks to read the cached zillow.csv first """
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_new_zillow_data()

def get_new_zillow_data():
    """Returns a dataframe of all 2017 properties that are Single Family Residential"""

    sql = '''
            SELECT *
            FROM properties_2017
            RIGHT JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
            WHERE propertylandusedesc = "Single Family Residential"
            AND predictions_2017.transactiondate LIKE "2017%%"
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL;
        '''
    df = pd.read_sql(sql, get_db_url('zillow'))
    return df

def wrangle_zillow():
    """
    Acquires Zillow data
    Handles nulls
    optimizes or fixes data types
    handles outliers w/ manual logic
    returns a clean dataframe
    """
    df = get_zillow_data()

    # df.to_csv("zillow.csv", index=False)

    return df





