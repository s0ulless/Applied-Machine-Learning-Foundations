"""
Evaluation (Polars)
R2
The Coefficient of Determination, R2, is a measure of how well the model fits the data. It is a value between 0 and 1.
It tells us how much of the variance in the target variable is predictable from the features.

A value of 0 means that the model explains none of the variability. A value of 1 means that the model explains all the
variability.

Note that it doesn't indicate whether a model is overfitting or underfitting the data.
"""

from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_pandas import get_pandas_df
from pipes_and_transformers_pandas import get_pipeline_pandas

# Gets Polars dataframe.
raw = get_pandas_df()

y = raw["price"]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

from catboost import CatBoostRegressor

cat = CatBoostRegressor()
cat_pipe = get_pipeline_pandas(cat)
cat_pipe.fit(X_train, y_train)

print(cat_pipe.score(X_test, y_test.to_numpy()))

# Mean Squared/ Absolute Error
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test, cat_pipe.predict(X_test)))

# rmse
# Unable to get this to work.
# mean_squared_error(y_test, cat_pipe.predict(X_test), squared=False)

# Alternative
import numpy as np

mse = mean_squared_error(y_test, cat_pipe.predict(X_test))
rmse = np.sqrt(mse)
print(rmse)

# absolute error
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, cat_pipe.predict(X_test))


# Compare lr model
from sklearn.linear_model import LinearRegression

lr =  LinearRegression()
lr_pipe = get_pipeline_pandas(lr)
lr_pipe.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, lr_pipe.predict(X_test)))


# Residual Plot
# make a residual plot
import matplotlib.pyplot as plt

ax = plt.scatter(cat_pipe.predict(X_test), 
    y_test - cat_pipe.predict(X_test), alpha=0.1)
# make labels not be scientific notation
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.ylim(-500_000, 500_000)
plt.xlabel('Predicted price')
plt.ylabel('Residual')
plt.title('Residual plot')
plt.show()

import pandas as pd

df = pd.DataFrame({
    "actual_price": y_test,
    "predicted_price": cat_pipe.predict(X_test)
})
df["residual"] = df["actual_price"] - df["predicted_price"]

# Scatter plot
ax = df.plot.scatter(
    x="predicted_price",
    y="residual",
    alpha=0.1
)

# Format axes with currency style
ax.xaxis.set_major_formatter('${:.0f}'.format)
ax.yaxis.set_major_formatter('${:.0f}'.format)

plt.show()

# Function to plot residual plot
def residuals_plot(model, X_train, y_train, X_test, y_test):
    # Predictions from the pipeline
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Build DataFrames for train and test residuals
    df_train = pd.DataFrame({
        "prediction": y_train_pred,
        "residual": y_train - y_train_pred,
        "type": "train"
    })

    df_test = pd.DataFrame({
        "prediction": y_test_pred,
        "residual": y_test - y_test_pred,
        "type": "test"
    })

    # Combine train and test residuals
    df = pd.concat([df_train, df_test], ignore_index=True)

    # Plot residuals
    plt.figure(figsize=(8, 6))
    colors = df["type"].map({"train": "blue", "test": "orange"})
    plt.scatter(df["prediction"], df["residual"], alpha=0.5, c=colors)
##    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot (Pipeline)")
##    plt.legend(["Zero line", "Train", "Test"])
    plt.show()

residuals_plot(cat_pipe, X_train, y_train, X_test, y_test)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_pandas(dt)
dt_pipe.fit(X_train, y_train)

residuals_plot(dt_pipe, X_train, y_train, X_test, y_test)
