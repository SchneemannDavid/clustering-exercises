







#### Josh's function for narrowing bad data results #### 
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