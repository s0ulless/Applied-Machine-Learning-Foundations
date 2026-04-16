"""
MLFlow Challenge & Solution

Reformat your notebook so that you can load the data and create an optimized random forest model in a single cell.
Then, use MLFlow to log the model and its parameters.
"""

# Initialisation
from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_polars import get_polars_df
from pipes_and_transformers_polars import get_pipeline_polars

# Gets Polars dataframe.
raw = get_polars_df()

y = raw.select("price")
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# Setting up RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf_pipe = get_pipeline_polars(rf)
rf_pipe.fit(X_train, y_train)

# Using MLFlow
import mlflow

model_info = mlflow.sklearn.log_model(rf_pipe, name='rf_pipe')

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(rf_pipe, "rf_pipe")
    run_id = run.info.run_id
    print("Run ID:", run_id)

model = mlflow.pyfunc.load_model(f"runs:/{run_id}/rf_pipe")
print(model.predict(X_test))
