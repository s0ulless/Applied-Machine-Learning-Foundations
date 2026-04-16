"""
Evaluate Random Forest Challenge & Solution
What is the mean squared error of the Random Forest model?
What is the R2 score?
What do these values tell us about the model?
"""

# Initialisation
from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_pandas import get_pandas_df
from pipes_and_transformers_pandas import get_pipeline_pandas

# Gets Polars dataframe.
raw = get_pandas_df()

y = raw["price"]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf_pipe = get_pipeline_pandas(rf)
rf_pipe.fit(X_train, y_train)

# mse
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, rf_pipe.predict(X_test)))

# r2
from sklearn.metrics import r2_score

print(r2_score(y_test, rf_pipe.predict(X_test)))
