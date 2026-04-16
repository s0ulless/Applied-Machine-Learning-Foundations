# Exploring Data with Polars
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn 
import catboost
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Method one
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
raw = pl.read_csv(file, new_columns=cols, skip_rows=31, has_header=False)

print("1a. type(raw)\n", type(raw))
print("1b. raw\n", raw)
print("1c. raw.describe()\n", raw.describe())
print("1d. raw.corr\n", raw.corr)
print("1d_ii. raw.corr()\n", raw.corr())
print("1e. raw.columns\n", raw.columns)
print("1f. raw.rows\n", raw.rows)
print("1e. raw.select('date_day', 'date_month', 'date_year')\n", raw.select('date_day', 'date_month', 'date_year'))
