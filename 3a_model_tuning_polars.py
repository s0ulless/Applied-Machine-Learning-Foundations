"""
Model Tuning (Polars)
Hyperparameters are the levers we can pull to adjust the behavior of a model.
They are set before the model is trained and remain constant during training.

Overfitting means that model is too complicated and that it's just memorising
the data. Underfitting means that model is too simple and it's unable to capture
the signal that's in the data.
"""

# Initialisation
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_polars import get_polars_df
from pipes_and_transformers_polars import get_pipeline_polars

# Gets Polars dataframe.
raw = get_polars_df()

y = raw.select("price")
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# Tuning Linear Regression
from sklearn.linear_model import Ridge

help(Ridge)

rr = Ridge()

rr_pipe = get_pipeline_polars(rr)
rr_pipe.fit(X_train, y_train)
print(rr_pipe.score(X_test, y_test))

# Sweeps through various values and keep the score
# Example param_range and scores from your loop
param_range = [0, .01, .05, .1, .5, 1, 2]
scores = []

for val in param_range:
    rr_pipe.set_params(model__alpha=val)
    rr_pipe.fit(X_train, y_train)
    scores.append(rr_pipe.score(X_test, y_test))

"""
Unable to get following code to work

# Our be score is at 0 (which is normal Linear Regression)
alpha = pl.DataFrame({'val': param_range,
              'scores': scores})
alpha.plot(x='val', y='scores')
"""

alpha = pl.DataFrame({'val': param_range, 'scores': scores}, strict=False)
alpha_pd = alpha.to_pandas()
alpha_pd.plot(x='val', y='scores')
plt.show()


# Tuning Decision Trees
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_polars(dt)
dt_pipe.fit(X_train, y_train)

"""
'model' is used instead of 'dt' because the pipeline process has been
moved to get_pipeline_polars in pipes_and_transformers_polars.py
"""
dt_pipe.named_steps['model']
help(dt_pipe.named_steps['model'])

from sklearn.model_selection import validation_curve
"""
'model__max_depth' is used instead of 'dt__max_depth' because the pipeline process has been
moved to get_pipeline_polars in pipes_and_transformers_polars.py
"""
param_range = range(1, 20)
train_scores, test_scores = validation_curve(
    dt_pipe, X_train, y_train, param_name="model__max_depth", param_range=param_range,
    scoring="neg_mean_squared_error", n_jobs=1)

# make a validation curve from train_scores and test_scores
import numpy as np

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with Decision Tree")
plt.xlabel("max_depth")
plt.ylabel("Score")
#plt.ylim(-1, 0)
lw = 2
plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                color="navy", lw=lw)

plt.fill_between(param_range, test_scores_mean - test_scores_std,   
                    test_scores_mean + test_scores_std, alpha=0.2,
                    color="navy", lw=lw)
plt.legend(loc="best")
plt.show()

# train dt_pipe with max_depth=8
dt8 = DecisionTreeRegressor(max_depth=8)
dt8_pipe = get_pipeline_polars(dt8)
dt8_pipe.fit(X_train, y_train)
print(dt8_pipe.score(X_test, y_test))

# dt8 mse
from sklearn.metrics import mean_squared_error
"""
unable to get mean_squared_error(y_test, dt8_pipe.predict(X_test), squared=False) to work
"""
y_pred = dt8_pipe.predict(X_test)
# Ensure both are 1-D float arrays
y_test = np.ravel(y_test).astype(float)
y_pred = np.ravel(y_pred).astype(float)

mse = np.mean((y_test - y_pred) ** 2)
print(mse)

# dt8 rmse
print(np.sqrt(mse))

# dt mse
y_pred = dt_pipe.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(mse)

# dt rmse
print(np.sqrt(mse))


# Grid Search
dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_polars(dt)
dt_pipe.fit(X_train, y_train)
# Store default decision tree score (withou optimised params) for comparison below.
dt_default_score = dt_pipe.score(X_test, y_test)
print(dt_default_score)

# use grid search on decision tree
from sklearn.model_selection import GridSearchCV
"""
'model__' is used instead of 'dt__' because the pipeline process has been
moved to get_pipeline_polars in pipes_and_transformers_polars.py
"""
param_grid = {
    'model__max_depth': [3, 6, 9],
    'model__min_samples_split': [10, 20, 100],
    'model__min_samples_leaf': [10, 20, 100],
}

grid_search = GridSearchCV(dt_pipe, param_grid, cv=5)#, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters as a result from Grid Search
print(grid_search.best_params_)

# Sets the best params using results from above
dt_pipe.set_params(**grid_search.best_params_)

dt_pipe.fit(X_train, y_train)
print(dt_pipe.score(X_test, y_test)) # Compare with dt_default_score
