# Model Creation (Polars)
from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_polars import get_polars_df
from pipes_and_transformers_polars import get_pipeline_polars

# Initialisation
# Gets Polars dataframe.
raw = get_polars_df()

y = raw.select("price")
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

# Dummy Model
from sklearn.dummy import DummyRegressor

dummy = DummyRegressor(strategy="mean")

# Gets pipe.
"""
Following code has been shifted to pipes_and_transformers_polars.py for reuse with other models.

input_model = Pipeline(steps=[('tweak', tweak_transformer),
                      ('zip_avg_price', ZipAvgPriceAdder()),
                      ('preprocessor', preprocessor),
                      ('input_model', input_model),
                      ])
"""

dummy_pipe = get_pipeline_polars(dummy)
dummy_pipe.fit(X_train, y_train)

print(dummy_pipe.score(X_test, y_test))
print(dummy_pipe)
print(dummy_pipe.predict(X_test))


# Linear Regression
from sklearn.linear_model import LinearRegression

lr =  LinearRegression()
lr_pipe = get_pipeline_polars(lr)
lr_pipe.fit(X_train, y_train)

print(lr_pipe.score(X_test, y_test))
print(lr_pipe.predict(X_test))


# Decision Trees
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
dt_pipe = get_pipeline_polars(dt)
dt_pipe.fit(X_train, y_train)

print(dt_pipe.score(X_test, y_test))
# Sets max_depth=1
dt_pipe.set_params(model__max_depth=1)
dt_pipe.fit(X_train, y_train)
print(dt_pipe.score(X_test, y_test))
# Sets max_depth=9
dt_pipe.set_params(model__max_depth=9)
dt_pipe.fit(X_train, y_train)
print(dt_pipe.score(X_test, y_test))


# CatBoost
from catboost import CatBoostRegressor

cat = CatBoostRegressor()
cat_pipe = get_pipeline_polars(cat)
cat_pipe.fit(X_train, y_train.to_numpy()[:,0])

print(cat_pipe.score(X_test, y_test.to_numpy()[:,0]))
