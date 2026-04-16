"""
Model Tuning (Polars)
Hyperparameters are the levers we can pull to adjust the behavior of a model.
They are set before the model is trained and remain constant during training.

Overfitting means that model is too complicated and that it's just memorising
the data. Underfitting means that model is too simple and it's unable to capture
the signal that's in the data.
"""

# Initialisation
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_pandas import get_pandas_df
from pipes_and_transformers_pandas import get_pipeline_pandas

# Gets Polars dataframe.
raw = get_pandas_df()

y = raw["price"]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

"""
Tuning CatBoost
Boosting - iterations (num_trees, n_estimators), learning_rate (eta), early_stopping_rounds

Tree based - depth (max_depth), grow_policy, min_child_samples (min_data_in_leaf), max_leaves (num_leaves)

Sampling - subsample, sampling_frequency, rsm (colsample_bylevel), random_strength, bagging_temperature

Regularization - l2_leaf_reg (reg_lambda), model_shrink_rate

Constraints - monotone_constraints, feature_weights
"""

# Takes quite awhile to run.
from catboost import CatBoostRegressor

help(CatBoostRegressor)

cr2 = CatBoostRegressor(iterations=3000, learning_rate=0.1, early_stopping_rounds=10)

X_train, X_test, y_train, y_test = train_test_split(raw.drop(columns=['price']), 
                                                    y, test_size=0.2, random_state=42)

cr2.fit(X_train, y_train, cat_features=['zipcode'], verbose=100, eval_set=(X_test, y_test))

# plot a validation curve tracking mse as the max_depth of the decision tree increases
##from sklearn.model_selection import validation_curve
##
##param_range = range(1, 10)
##train_scores, test_scores = validation_curve(
##    cr2, X_train, y_train, param_name="max_depth", 
##    param_range=param_range,
##    scoring="neg_mean_squared_error", n_jobs=1)
"""    
Unable to get the following arguments to work.
    fit_params=dict(early_stopping_rounds=10, 
                    eval_set=(X_test, y_test))
"""

# Codex AI recommended fix
from sklearn.model_selection import validation_curve

param_range = np.array(list(range(1, 10)), dtype=float)

train_scores, test_scores = validation_curve(
    cr2,
    X_train,
    y_train,
    param_name="max_depth",
    param_range=range(1, 10),
    scoring="neg_mean_squared_error",
    n_jobs=1,
    params={
        # only if cr2.fit supports these
        "early_stopping_rounds": 10,
        "eval_set": [(X_test, y_test)],
    },
)


# make a validation curve from train_scores and test_scores
train_scores_mean = np.asarray(np.mean(train_scores, axis=1), dtype=float)
train_scores_std = np.asarray(np.std(train_scores, axis=1), dtype=float)
test_scores_mean = np.asarray(np.mean(test_scores, axis=1), dtype=float)
test_scores_std = np.asarray(np.std(test_scores, axis=1), dtype=float)

plt.title("Validation Curve with CatBoost")
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

# set max_depth to 4
cr2_4 = CatBoostRegressor(iterations=3000, learning_rate=0.1,
                                max_depth=4)

X_train, X_test, y_train, y_test = train_test_split(raw.drop(columns=['price']), y, 
                                                    test_size=0.2, random_state=42)

cr2_4.fit(X_train, y_train, cat_features=['zipcode'], verbose=100,
        early_stopping_rounds=10, eval_set=(X_test, y_test))
print(cr2_4.score(X_test, y_test))
