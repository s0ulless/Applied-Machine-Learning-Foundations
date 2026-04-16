# Custom transformer maps a zip code to the average price of that zip code.
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    # Assume X is a pandas dataframe.
    # Group X by the zip code, then aggregate that to get the average price.   
    def fit(self, X, y=None):
        self.zip_avg_price = (X.group_by('zipcode')
                              .agg(zip_mean=pl.col('price').mean()))
        return self

    # Transform data, where X is the dataframe.
    # Here we are going to add the zip price average information on it.
    def transform(self, X, y=None):
        return X.join(self.zip_avg_price, on='zipcode')
