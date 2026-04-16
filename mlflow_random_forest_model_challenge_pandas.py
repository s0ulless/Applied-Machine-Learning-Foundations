"""
Challenge
Load the data, create an optimized RandomForest model in one cell,
and log the model and its parameters with MLflow.
"""

import pandas as pd
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

# --- Load data with pandas ---
url = 'https://www.openml.org/data/download/22044765/dataset'
cols = ['id','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view',
        'condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated',
        'zipcode','lat','long','sqft_living15','sqft_lot15','date_year','date_month','date_day']

raw = pd.read_csv(url, skiprows=31, header=None, names=cols)

# --- Feature engineering ---
def tweak_housing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['zipcode'] = df['zipcode'].astype(str).astype('category')
    df['date'] = pd.to_datetime(dict(year=df['date_year'],
                                     month=df['date_month'],
                                     day=df['date_day']))
    df['yr_renovated'] = df['yr_renovated'].replace(0, np.nan)
    return df[['id','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
               'waterfront','view','condition','grade','sqft_above','sqft_basement',
               'yr_built','yr_renovated','zipcode','lat','long','sqft_living15',
               'sqft_lot15','date']]

class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        self.zip_avg_price = X.groupby('zipcode', observed=True)['price'].mean().reset_index()
        self.zip_avg_price.rename(columns={'price':'zip_mean'}, inplace=True)
        return self
    def transform(self, X: pd.DataFrame, y=None):
        return X.merge(self.zip_avg_price, on='zipcode', how='left')

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

tweak_transformer = FunctionTransformer(tweak_housing)

# Optimized RandomForest
rf = RandomForestRegressor(max_depth=9)

y = raw[['price']]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

rf_pipe = Pipeline(steps=[
    ('tweak', tweak_transformer),
    ('zip_avg_price', ZipAvgPriceAdder()),
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
