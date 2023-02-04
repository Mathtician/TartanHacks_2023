import numpy as np
import pandas as pd
import torch

inFile = "../data extraction/prepared_data.csv"
df = pd.read_csv(inFile)
dfNums = df.copy()

categories = []

for i, col in enumerate(df.columns.values):
    if (df.dtypes[col] != np.float64):
        df[col] = df[col].astype("category")
        dfNums = pd.get_dummies(dfNums, columns=[col], dtype=np.float64)
        labels = df[col].cat.categories.tolist()
        categories.append({
            "name": col,
            "labels": labels
        })
        #featureToCol.extend([i]*len(labels))
        #print(col, numUnique)
    #else:
        #featureToCol.append(i)

numCols = len(df.columns.values)

i = numCols-len(categories)
featureToCol = list(range(i))
for cat in categories:
    featureToCol.extend([i]*len(cat["labels"]))
    i += 1

numFeatures = len(featureToCol)

#print(categories)
#print(featureToCol)

#print(numCols, numFeatures)

def maskData(data, p=0.1):
    batchSize = data.shape[0]
    mask = np.random.random((batchSize, numCols)) < p
    maskedData = np.concatenate((data, mask), axis = 1)
    #print(mask)
    for f in range(numFeatures):
        c = featureToCol[f]
        maskedData[:, f] *= mask[:, c]
    #print(data)
    return maskedData

dfNumpy = dfNums.to_numpy()
batch = dfNumpy[:10]
print(featureToCol)
print(batch)
maskedBatch = maskData(batch)
print(maskedBatch)
