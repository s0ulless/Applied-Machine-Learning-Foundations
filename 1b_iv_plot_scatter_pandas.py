import get_dataset_pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Gets Pandas dataframe.
raw = get_dataset_pandas.get_pandas_df()
# --- Lat/long scatter plot (all prices) ---
df_scatter = raw.sort_values('price')

# --- Lat/long scatter plot (all prices) ---
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_scatter['long'], df_scatter['lat'],
                      c=df_scatter['price'], s=1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label='Price')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Lat/Long Scatter Plot Colored by Price")

# --- Lat/long scatter plot (price > $1M) ---
df_high = raw[raw['price'] > 1_000_000].sort_values('price')

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_high['long'], df_high['lat'],
                      c=df_high['price'], s=1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label="Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Lat/Long Scatter Plot (Price > $1M)")
plt.show()
