import pandas as pd
import numpy as np

# Data preprocessing/cleaning
def tweak_housing(df: pd.DataFrame) -> pd.DataFrame:
    # Cast zipcode to string then categorical
    df['zipcode'] = df['zipcode'].astype(str).astype('category')
    
    # Build a proper datetime column from year, month, day
    df['date'] = pd.to_datetime(
        dict(year=df['date_year'], month=df['date_month'], day=df['date_day'])
    )
    
    # Replace 0 with NaN in yr_renovated
    df['yr_renovated'] = df['yr_renovated'].replace(0, np.nan)
    
    # Select the desired columns
    df = df[['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
             'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
             'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
             'sqft_lot15', 'date']]
    
    return df
