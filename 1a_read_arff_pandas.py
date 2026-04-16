### Exploring Data with Pandas
import pandas as pd
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import catboost
import warnings

from scipy.io import arff

warnings.filterwarnings('ignore')

### Method one
##url = 'https://www.openml.org/data/download/22044765/dataset'
# File from URL downloaded and saved as dataset.arff.
file = 'dataset.arff'
cols = ['id', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 
        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
        'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'date_year', 'date_month', 'date_day']

"""
skiprows=31 skips the first 31 lines of the arff dataset file,
(which contains @RELATION, @ATTRIBUTE, etc. i.e., its metadata)
thus allowing it to be read like raw csv file
"""
raw = pd.read_csv(file, names=cols, skiprows=31, header=None)

print("1a. type(raw)\n", type(raw))
print("1b. raw\n", raw)
print("1c. raw.dtypes\n", raw.dtypes)
print("1d. raw.describe\n", raw.describe)
print("1e. raw.corr()\n", raw.corr())
print("1f. raw.columns\n", raw.columns)


### Method 2
# Load ARFF file
data, meta = arff.loadarff(file)

# Convert to DataFrame
df = pd.DataFrame(data)

print("2a. type(df)\n", type(df))
print("2b. df\n", df)
print("2c. df.dtypes\n", df.dtypes)
print("2d. df.describe\n", df.describe)
print("2e. df.corr()\n", df.corr())
print("2f. df.columns\n", df.columns)

### Miscellaneous - Extracting column names
# Extract column names directly from ARFF metadata
print("2g. meta.names()\n", meta.names())
# Extract column names directly from 'data' 
print("2h. data.dtype.names\n", data.dtype.names)

