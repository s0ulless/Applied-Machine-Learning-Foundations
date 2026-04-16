# The difference between sklearn pipelines and transformers is 
# that a pipeline is a sequence of steps. A transformer transforms
# the data, and a pipeline is a sequence of transformers.
# A ColumnTransformer applies multiple transformers to different
# columns of the input data.
import polars as pl
import polars.selectors as cs
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import set_config
# Custom imports
from get_dataset_polars import get_polars_df
from process_data_polars import tweak_housing

set_config(transform_output='polars')

# Gets Polars dataframe
raw = get_polars_df()
# See what the numeric columns are.
print(tweak_housing(raw).select(cs.numeric()).columns)

"""
Imagine comparing number of bedrooms or bathrooms to square feet.
If these are not standardised, some algorithms may pay more attention to square feet
since the numbers are much larger than the no. of rooms.

StandardScaler's fit calculates the mean and standard deviation for each numeric column.
These values are stored internally in the scaler object, std.mean_, std.scale_ etc.

transform produces standardised data with mean that approximates to 0 and varience that 
approximates to 1.

The following codes has additional print statements to work with Python's IDLE.
"""
# Define numeric features
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living']
# Apply StandardScaler
std = StandardScaler()
std.fit_transform(tweak_housing(raw).select(numeric_features))

# Build pipeline
num_pipeline = Pipeline([
     ('std', StandardScaler())])

# Fit and transform
num_pipe = num_pipeline.fit_transform(
    tweak_housing(raw)
    .select(numeric_features)
)
print(num_pipe)
# Add another step
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), # if a value is missing, add the median value
    ('std', StandardScaler())])

# Fit and transform
num_p = num_pipeline.fit_transform(
    tweak_housing(raw)
    .select(numeric_features)
)
print(num_p)
"""
If we train it on a dataset that has certain categories and when we try
to do prediction, if it comes across a new category, we ignore it.

max_categories sets maximum no. of columns.
"""
cat_features = ['zipcode']

ohe = OneHotEncoder(handle_unknown='ignore',
                    sparse_output=False, max_categories=10)

# Fit and transform
ohe_f = ohe.fit_transform(
    tweak_housing(raw)
    .select(cat_features)
)
print(ohe_f)

# Transformer from a function.
tweak_transformer = FunctionTransformer(tweak_housing)
print(tweak_transformer.fit_transform(raw))

# Column transformer
categorical_features = ['zipcode']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Column Transformer lets us apply specific transformations to certain columns.
ct = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False), categorical_features)])

ct_f = ct.fit_transform(
    tweak_housing(raw)
    .select([*numeric_features, *cat_features])
)
print(ct_f)

# Custom transformer that maps a zip code to the average price of that zip code.
class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # assume X is a polars dataframe
        self.zip_avg_price = (X
                              .group_by('zipcode')
                              .agg(zip_mean=pl.col('price').mean())
        )
        return self
    
    def transform(self, X, y=None):
        return X.join(self.zip_avg_price, on='zipcode')

zip_adder = ZipAvgPriceAdder()
zip_f = zip_adder.fit_transform(raw.select(['zipcode', 'price']))
print(zip_f)
