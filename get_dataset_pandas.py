### Exploring Data with Pandas
import pandas as pd
import warnings

from scipy.io import arff

warnings.filterwarnings('ignore')

def get_pandas_df ():
    file = 'dataset.arff'
    cols = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day']

    return pd.read_csv(file, names=cols, skiprows=31, header=None)
