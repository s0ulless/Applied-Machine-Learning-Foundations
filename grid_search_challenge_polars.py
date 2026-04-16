"""
Grid Search Challenge & Solution

Do a grid search to find the best depth for the random forest model.
What is the best depth? What is the score of the model with the best depth?
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

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf_pipe = get_pipeline_polars(rf)
rf_pipe.fit(X_train, y_train)
print(rf_pipe.score(X_test, y_test)) # Compare this default score with the optimised one below.

# Search for best parameters.
from sklearn.model_selection import GridSearchCV
"""
'model__max_depth' is used instead of 'rf__max_depth' because the pipeline process has been
moved to get_pipeline_polars in pipes_and_transformers_polars.py
"""
param_grid = {
    'model__max_depth': [3, 4, 6, 7, 9],
}

grid_search = GridSearchCV(rf_pipe, param_grid, cv=5) #, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

# Run with max_depth=9
rf = RandomForestRegressor(max_depth=9)

rf_pipe = get_pipeline_polars(rf)
rf_pipe.fit(X_train, y_train)

print(rf_pipe.score(X_test, y_test)) # Compare with the default score above.
