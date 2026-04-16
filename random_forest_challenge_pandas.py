# Random Forest challenge.
from sklearn.model_selection import train_test_split
# Custom import
from get_dataset_pandas import get_pandas_df
from pipes_and_transformers_pandas import get_pipeline_pandas

# Gets Pandas dataframe.
raw = get_pandas_df()

y = raw["price"]
X_train, X_test, y_train, y_test = train_test_split(raw, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf_pipe = get_pipeline_pandas(rf)
rf_pipe.fit(X_train, y_train)

print(rf_pipe.score(X_test, y_test))
