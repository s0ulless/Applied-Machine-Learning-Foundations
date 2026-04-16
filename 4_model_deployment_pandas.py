"""
Model Deployment
End to end notebook (Pandas version)
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# --- Tweaking function for housing dataset ---
def tweak_housing(df):
    df = df.copy()
    df['zipcode'] = df['zipcode'].astype(str).astype('category')
    df['date'] = pd.to_datetime(dict(year=df['date_year'],
                                     month=df['date_month'],
                                     day=df['date_day']))
    # Replace 0 with np.nan (not pd.NA)
    df['yr_renovated'] = df['yr_renovated'].replace(0, np.nan)
    return df[['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
               'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above',
               'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
               'sqft_living15', 'sqft_lot15', 'date']]

# --- Numeric and categorical features ---
numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                    'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                    'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long',
                    'sqft_living15', 'sqft_lot15', 'zip_mean']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['zipcode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

# --- Custom transformer to add zipcode average price ---
class ZipAvgPriceAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.zip_avg_price = X.groupby('zipcode')['price'].mean().reset_index(name='zip_mean')
        return self
    
    def transform(self, X, y=None):
        return X.merge(self.zip_avg_price, on='zipcode', how='left')

# --- Load dataset (Pandas) ---
url = 'https://www.openml.org/data/download/22044765/dataset'
cols = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
        'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15',
        'sqft_lot15', 'date_year', 'date_month', 'date_day']

raw = pd.read_csv(url, names=cols, skiprows=31)

# --- Train/test split ---
y = raw['price']
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# --- Build pipeline ---
lr = LinearRegression()
tweak_transformer = FunctionTransformer(tweak_housing)

lr_pipe = Pipeline(steps=[
    ('tweak', tweak_transformer),
    ('zip_avg_price', ZipAvgPriceAdder()),
    ('preprocessor', preprocessor),
    ('lr', lr)
])

# --- Fit and evaluate ---
lr_pipe.fit(X_train, y_train)
print("1. lr_pipe.score(X_test, y_test)", lr_pipe.score(X_test, y_test))
