# Exploring Data with Polars
import polars as pl
import warnings

warnings.filterwarnings('ignore')

def get_polars_df():
    file = 'dataset.arff'
    cols = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
            'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
            'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day']

    return pl.read_csv(file, new_columns=cols, skip_rows=31, has_header=False)
