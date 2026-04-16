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
from get_dataset_polars import get_polars_df
from pipes_and_transformers_polars import get_pipeline_polars

# Gets Polars dataframe.
raw = get_polars_df()

y = raw.select("price")
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

from catboost import CatBoostRegressor

cat = CatBoostRegressor()
cat_pipe = get_pipeline_polars(cat)
cat_pipe.fit(X_train, y_train.to_numpy()[:,0])

print(cat_pipe.score(X_test, y_test.to_numpy()[:,0]))

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
lr_pipe = get_pipeline_polars(lr)
lr_pipe.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, lr_pipe.predict(X_test)))


# Residual Plot
# make a residual plot
import matplotlib.pyplot as plt

ax = plt.scatter(cat_pipe.predict(X_test), 
    y_test.to_series().to_numpy() - cat_pipe.predict(X_test), alpha=0.1)
# make labels not be scientific notation
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.ylim(-500_000, 500_000)
plt.xlabel('Predicted price')
plt.ylabel('Residual')
plt.title('Residual plot')
plt.show()

# Plot with ploars
(y_test
 .with_columns(predicted_price=cat_pipe.predict(X_test),
   residual=y_test.to_series().to_numpy() - cat_pipe.predict(X_test))
 .plot.scatter('predicted_price', 'residual')
 )

# Function to plot residual plot
import polars as pl

def residuals_plot(model, X_train, y_train, X_test, y_test):
    df = (
        y_test.with_columns(
            prediction=model.predict(X_test),
            residual=y_test.to_series().to_numpy() - model.predict(X_test),
            type=pl.lit('test')
        )
        .vstack(
            y_train.with_columns(
                prediction=model.predict(X_train),
                residual=y_train.to_series().to_numpy() - model.predict(X_train),
                type=pl.lit('train')
            )
        )
        .reverse()
    )

    # Use matplotlib directly for plotting
    plt.scatter(df['prediction'], df['residual'], alpha=0.5)
##    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.show()

residuals_plot(cat_pipe, X_train, y_train, X_test, y_test)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_polars(dt)
dt_pipe.fit(X_train, y_train)

residuals_plot(dt_pipe, X_train, y_train, X_test, y_test)
