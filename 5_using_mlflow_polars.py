"""
Using MLFlow
Going to show how to persist and load a model, but can also:

Start a endpoint to serve predictions
Build a Docker image
"""

# Initialisation
from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_polars import get_polars_df
from pipes_and_transformers_polars import get_pipeline_polars

# Initialisation
# Gets Polars dataframe.
raw = get_polars_df()

y = raw.select("price")
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# Linear Regression for use with MLFlow e.g.
from sklearn.linear_model import LinearRegression

lr =  LinearRegression()
lr_pipe = get_pipeline_polars(lr)
lr_pipe.fit(X_train, y_train)

import mlflow
print(mlflow.__version__)

model_info = mlflow.sklearn.log_model(lr_pipe, name='lr_pipe')

print(model_info.artifact_path)

import subprocess

subprocess.run("cmd /c tree", shell=True)

# Log and load model correctly
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(lr_pipe, "lr_pipe")
    run_id = run.info.run_id
    print("Run ID:", run_id)
    
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/lr_pipe")
print(model)

print(model.predict(X_test))
