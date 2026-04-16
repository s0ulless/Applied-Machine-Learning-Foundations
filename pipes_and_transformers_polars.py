# Consolidates the required components for setting up pipelines.
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# Custom imports
from get_dataset_polars import get_polars_df
from process_data_polars import tweak_housing
# Custom transformer, maps a zip code to the average price of that zip code.
from ZipAvgPriceAdderPl import ZipAvgPriceAdder

# Gets Polars dataframe
raw = get_polars_df()

# Numeric features, i.e., columns.
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
                    'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
                    'lat', 'long', 'sqft_living15', 'sqft_lot15', 'zip_mean']
# Categorial features, i.e., non-numeric.
categorical_features = ['zipcode']

# Transformer from a function
tweak_transformer = FunctionTransformer(tweak_housing)

# This pipeline is for imputing and then standardising the numbers.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Setting up preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False), categorical_features)])

# Returns user pipeline, based on model input.
# This can be anything from Dummy, Decision Tree to Linear Regression etc.
"""
To set pipeline parems, for e.g.:
instead of dt_pipe.set_params(dt__max_depth=1),
use dt_pipe.set_params(model__max_depth=1)
"""
def get_pipeline_polars(model):
    pipe = Pipeline(steps=[('tweak', tweak_transformer),
                           ('zip_avg_price', ZipAvgPriceAdder()), # adds price_zip_mean
                           ('preprocessor', preprocessor),
                           ('model', model),
                           ])
    return pipe
