import get_dataset_polars
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

# Get Polars DataFrame and converts it to Pandas
raw = get_dataset_polars.get_polars_df()
df_scatter = raw.sort('price').to_pandas()

# --- Lat/long scatter plot (all prices) ---
df_scatter = raw.sort('price').to_pandas()

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_scatter['long'], df_scatter['lat'],
                      c=df_scatter['price'], s=1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label="Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Lat/Long Scatter Plot Colored by Price")
plt.show()

# --- Lat/long scatter plot (price > $1M) ---
df_high = raw.filter(pl.col('price') > 1_000_000).sort('price').to_pandas()
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_high['long'], df_high['lat'],
                      c=df_high['price'], s=1, alpha=0.5, cmap='viridis')
plt.colorbar(scatter, label="Price")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Lat/Long Scatter Plot (Price > $1M)")
plt.show()
