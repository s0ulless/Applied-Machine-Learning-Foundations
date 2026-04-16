"""
Model Tuning (Polars)
Hyperparameters are the levers we can pull to adjust the behavior of a model.
They are set before the model is trained and remain constant during training.

Overfitting means that model is too complicated and that it's just memorising
the data. Underfitting means that model is too simple and it's unable to capture
the signal that's in the data.
"""

# Initialisation
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_pandas import get_pandas_df
from pipes_and_transformers_pandas import get_pipeline_pandas

# Gets Polars dataframe.
raw = get_pandas_df()

y = raw["price"]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# Tuning Linear Regression
from sklearn.linear_model import Ridge

help(Ridge)

rr = Ridge()

rr_pipe = get_pipeline_pandas(rr)
rr_pipe.fit(X_train, y_train)
print(rr_pipe.score(X_test, y_test))

# Sweeps through various values and keep the score
from sklearn.model_selection import validation_curve

# Example param_range and scores from your loop
param_range = [0, .01, .05, .1, .5, 1, 2]
scores = []

for val in param_range:
    rr_pipe.set_params(model__alpha=val)
    rr_pipe.fit(X_train, y_train)
    scores.append(rr_pipe.score(X_test, y_test))

alpha = pd.DataFrame({'val': param_range, 'scores': scores})
alpha.plot(x='val', y='scores')
plt.show()


# Tuning Decision Trees
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_pandas(dt)
dt_pipe.fit(X_train, y_train)

"""
'model' is used instead of 'dt' because the pipeline process has been
moved to get_pipeline_polars in pipes_and_transformers_pandas.py
"""
dt_pipe.named_steps['model']
help(dt_pipe.named_steps['model'])

"""
'model__max_depth' is used instead of 'dt__max_depth' because the pipeline process has been
moved to get_pipeline_pandas in pipes_and_transformers_pandas.py
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
dt8_pipe = get_pipeline_pandas(dt8)
dt8_pipe.fit(X_train, y_train)
print(dt8_pipe.score(X_test, y_test))

# dt8 mse
from sklearn.metrics import mean_squared_error
"""
unable to get mean_squared_error(y_test, dt8_pipe.predict(X_test), squared=False) to work
"""
y_pred = dt8_pipe.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(mse)

# dt8 rmse
print(np.sqrt(mse))

print(dt_pipe.score(X_test, y_test))

# dt mse
"""
unable to get mean_squared_error(y_test, dt_pipe.predict(X_test), squared=False) to work
"""
y_pred = dt_pipe.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
print(mse)

# dt rmse
print(np.sqrt(mse))


"""
Tuning CatBoost
Boosting - iterations (num_trees, n_estimators), learning_rate (eta), early_stopping_rounds

Tree based - depth (max_depth), grow_policy, min_child_samples (min_data_in_leaf), max_leaves (num_leaves)

Sampling - subsample, sampling_frequency, rsm (colsample_bylevel), random_strength, bagging_temperature

Regularization - l2_leaf_reg (reg_lambda), model_shrink_rate

Constraints - monotone_constraints, feature_weights
"""
# Takes quite awhile to run.
##from catboost import CatBoostRegressor
##
##help(CatBoostRegressor)
##
##cr2 = CatBoostRegressor(iterations=3000, learning_rate=0.1, early_stopping_rounds=10)
##
##X_train, X_test, y_train, y_test = train_test_split(raw.drop(columns=['price']), 
##                                                    y, test_size=0.2, random_state=42)
##
##cr2.fit(X_train, y_train, cat_features=['zipcode'], verbose=100, eval_set=(X_test, y_test))
##
### plot a validation curve tracking mse as the max_depth of the decision tree increases
##from sklearn.model_selection import validation_curve
##
##param_range = range(1, 10)
##train_scores, test_scores = validation_curve(
##    cr2, X_train, y_train, param_name="max_depth", 
##    param_range=param_range,
##    scoring="neg_mean_squared_error", n_jobs=1)
##"""    
##Unable to get the following arguments to work.
##    fit_params=dict(early_stopping_rounds=10, 
##                    eval_set=(X_test, y_test))
##"""
##
### make a validation curve from train_scores and test_scores
##train_scores_mean = np.mean(train_scores, axis=1)
##train_scores_std = np.std(train_scores, axis=1)
##test_scores_mean = np.mean(test_scores, axis=1)
##test_scores_std = np.std(test_scores, axis=1)
##
##plt.title("Validation Curve with CatBoost")
##plt.xlabel("max_depth")
##plt.ylabel("Score")
###plt.ylim(-1, 0)
##lw = 2
##plt.plot(param_range, train_scores_mean, label="Training score",
##             color="darkorange", lw=lw)
##plt.fill_between(param_range, train_scores_mean - train_scores_std,
##                 train_scores_mean + train_scores_std, alpha=0.2,
##                 color="darkorange", lw=lw)
##plt.plot(param_range, test_scores_mean, label="Cross-validation score",
##                color="navy", lw=lw)
##
##plt.fill_between(param_range, test_scores_mean - test_scores_std,   
##                    test_scores_mean + test_scores_std, alpha=0.2,
##                    color="navy", lw=lw)
##plt.legend(loc="best")
##plt.show()

# set max_depth to 4
cr2_4 = CatBoostRegressor(iterations=3000, learning_rate=0.1,
                                max_depth=4)

X_train, X_test, y_train, y_test = train_test_split(raw.drop(columns=['price']), y, 
                                                    test_size=0.2, random_state=42)

cr2_4.fit(X_train, y_train, cat_features=['zipcode'], verbose=100,
        early_stopping_rounds=10, eval_set=(X_test, y_test))
print(cr2_4.score(X_test, y_test))


# Grid Search
dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_pandas(dt)
dt_pipe.fit(X_train, y_train)
# Store default decision tree score (withou optimised params) for comparison below.
dt_default_score = dt_pipe.score(X_test, y_test)
print(dt_default_score)

# use grid search on decision tree
from sklearn.model_selection import GridSearchCV
"""
'model__' is used instead of 'dt__' because the pipeline process has been
moved to get_pipeline_pandas in pipes_and_transformers_pandas.py
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
