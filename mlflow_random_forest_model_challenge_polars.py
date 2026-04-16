"""
Challenge
Load the data with Polars, create an optimized RandomForest model in one cell,
and log the model and its parameters with MLflow.
"""

import polars as pl
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import mlflow

# --- Load data with Polars ---
url = 'https://www.openml.org/data/download/22044765/dataset'
cols = ['id','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
        'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
        'zipcode','lat','long','sqft_living15','sqft_lot15','date_year','date_month','date_day']

raw = pl.read_csv(url, new_columns=cols, skip_rows=31, has_header=False)

# --- Feature engineering ---
def tweak_housing(df: pl.DataFrame) -> pl.DataFrame:
    df = (df
          .with_columns([
              pl.col('zipcode').cast(pl.String).cast(pl.Categorical),
              pl.date(pl.col('date_year'), pl.col('date_month'), pl.col('date_day')).alias('date'),
              pl.col('yr_renovated').replace(0, None)
          ])
          .select(['id','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                   'waterfront','view','condition','grade','sqft_above','sqft_basement',
                   'yr_built','yr_renovated','zipcode','lat','long','sqft_living15',
                   'sqft_lot15','date'])
    )
    return df

class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def fit(self, X: pl.DataFrame, y=None):
        self.zip_avg_price = (X.group_by('zipcode')
                                .agg(pl.col('price').mean().alias('zip_mean')))
        return self
    def transform(self, X: pl.DataFrame, y=None):
        return X.join(self.zip_avg_price, on='zipcode')

# Convert Polars → pandas for scikit-learn preprocessing
def to_pandas(df: pl.DataFrame):
    return df.to_pandas()

tweak_transformer = FunctionTransformer(tweak_housing)
zip_avg_price_adder = ZipAvgPriceAdder()
pandas_transformer = FunctionTransformer(to_pandas)

# --- Pipeline setup ---
numeric_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
                    'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
                    'lat','long','sqft_living15','sqft_lot15','zip_mean']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['zipcode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)])

rf = RandomForestRegressor(max_depth=9)

y = raw.select('price')
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

rf_pipe = Pipeline(steps=[
    ('tweak', tweak_transformer),
    ('zip_avg_price', zip_avg_price_adder),
    ('to_pandas', pandas_transformer),   # convert Polars → pandas
    ('preprocessor', preprocessor),
    ('rf', rf),
])

# --- Train and evaluate ---
rf_pipe.fit(X_train, y_train)
print("RandomForest R^2 score:", rf_pipe.score(X_test, y_test))

# --- Log with MLflow ---
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(rf_pipe, "rf_pipe")
    mlflow.log_params({"n_estimators":200, "max_depth":12, "random_state":42})
    print("Run ID:", run.info.run_id)

# Load back
model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/rf_pipe")
print("Model loaded successfully")
print(model.predict(X_test))
