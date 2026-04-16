import get_dataset_polars
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

# Get Polars DataFrame
raw = get_dataset_polars.get_polars_df()

# Group by date_month and zipcode, compute mean price
df_grouped = (
    raw.group_by(["date_month", "zipcode"])
       .agg(pl.col("price").mean())
)

# Convert grouped Polars result to NumPy arrays for plotting
date_month = df_grouped["date_month"].to_numpy()
zipcode = df_grouped["zipcode"].to_numpy()
price = df_grouped["price"].to_numpy()

# Seaborn still expects a tidy DataFrame-like structure,
# so easiest is to convert just this grouped result to Pandas:
df_grouped_pd = df_grouped.to_pandas()

# Line plot
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_grouped_pd, x="date_month", y="price", hue="zipcode")
plt.title("Average Price by Month and Zipcode")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.legend(title="Zipcode")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
