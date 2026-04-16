"""
Challenge

Make a plot to explore the relationship between the number of bedrooms
and the price of the house.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
# Custom imports
from get_dataset_pandas import get_pandas_df
from process_data_pandas import tweak_housing

# Gets Polars dataframe.
df = tweak_housing(get_pandas_df())

# Scatter plot
ax = df.plot.scatter(x="bedrooms", y="price", alpha=.01)
# Format y-axis labels as integers
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

# For Python's IDLE.
plt.show()
