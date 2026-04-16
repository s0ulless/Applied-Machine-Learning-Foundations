# Random Forest model challenge.
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

print(rf_pipe.score(X_test, y_test))
