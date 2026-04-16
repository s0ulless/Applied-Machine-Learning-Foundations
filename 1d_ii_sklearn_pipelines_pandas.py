# Full Sklearn Pipelines E.g.
# The difference between sklearn pipelines and transformers is 
# that a pipeline is a sequence of steps. A transformer transforms
# the data, and a pipeline is a sequence of transformers.
# A ColumnTransformer applies multiple transformers to different
# columns of the input data.
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
# Custom imports
from get_dataset_pandas import get_pandas_df
from process_data_pandas import tweak_housing

# Gets Pandas dataframe.
raw = get_pandas_df()

# make the pipeline
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
                    'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zip_mean']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['zipcode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False), categorical_features)])

tweak_transformer = FunctionTransformer(tweak_housing)

# Custom transformer that maps a zip code to the average price of that zip code.
class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self):
            pass

    # Assume X is a pandas dataframe.
    # Group X by the zip code, then aggregate that to get the average price.   
    def fit(self, X, y=None):
        self.zip_avg_price = X.groupby('zipcode')['price'].mean().reset_index()
        return self
    
    # Transform data, where X is the dataframe.
    # Here we are going to add the zip price average information on it.
    def transform(self, X, y=None):
        return (X.merge(self.zip_avg_price, on='zipcode', suffixes=('', '_zip_mean'))
                .rename(columns={'price_zip_mean':'zip_mean'}))

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipe = Pipeline(steps=[('tweak', tweak_transformer),
                      ('zip_avg_price', ZipAvgPriceAdder()),
                      ('preprocessor', preprocessor),
                      ])

X = raw #.drop('price')
y = raw['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe_f = pipe.fit_transform(raw, raw['price'])
# For Python's IDLE
print(pipe_f)
print(pipe)
