import get_dataset_polars
import matplotlib.pyplot as plt
import seaborn as sns

# Gets Polars dataframe and converts it to Pandas for plotting heatmap.
raw = get_dataset_polars.get_polars_df().to_pandas()

# Scatter plot: sqft_living vs price
raw.plot.scatter(x='sqft_living', y='price', alpha=0.1)
plt.show()
