# converting normalized data to user-friendly statistics
import numpy as np
import pandas as pd

# sample just for testing
numerical_data = pd.read_csv('prepared_data.csv')
row_seven = numerical_data.iloc[0]
row_seven = row_seven.copy()

mean_stdevs = pd.read_csv('normalized_mean_stdev.csv', index_col='variable')
for i in mean_stdevs.index:
    mean = mean_stdevs.loc[i][0]
    stdev = mean_stdevs.loc[i][1]
    row_seven[i] = row_seven[i] * stdev + mean
print(row_seven)
    