import get_dataset_pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Gets Pandas dataframe.
df = get_dataset_pandas.get_pandas_df()
# Correlation heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(20, 10))
sns.heatmap(corr, cmap="RdBu", vmin=-1, vmax=1, annot=True) #-1 will be red, 1 will be blue
plt.title("Correlation Heatmap")
plt.show()
