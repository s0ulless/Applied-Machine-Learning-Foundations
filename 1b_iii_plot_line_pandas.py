import get_dataset_pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Gets Pandas dataframe.
raw = get_dataset_pandas.get_pandas_df()

# Group by date_month and zipcode, compute mean price
df = (raw
      .groupby(['date_month', 'zipcode'], as_index=False)['price']
      .mean())

# Line plot with seaborn
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='date_month', y='price', hue='zipcode')
plt.title("Average Price by Month and Zipcode")
plt.xlabel("Month")
plt.ylabel("Average Price")
plt.legend(title="Zipcode")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
