import numpy as np
import pandas as pd
from torch import nn

class CatInfo:

    def __init__(self):
        self.categories = []
        self.numCols = 0
        self.numVars = 0
        self.featureToCol = []
        self.numFeatures = 0
        self.numInputs = 0

class CSVData:
    
    def __init__(self, inFile):
        self.rawData = pd.read_csv(inFile)
        dfNums = self.rawData.copy()

        self.catInfo = CatInfo()
        
        for i, col in enumerate(self.rawData.columns.values):
            if self.rawData.dtypes[col] != np.float64:
                self.rawData[col] = self.rawData[col].astype("category")
                dfNums = pd.get_dummies(dfNums, columns=[col], dtype=np.float64)
                labels = self.rawData[col].cat.categories.tolist()
                self.catInfo.categories.append({
                    "name": col,
                    "labels": labels
                })

        # Number of variables+categories
        self.catInfo.numCols = len(self.rawData.columns.values)
        # Number of variables (floats)
        self.catInfo.numVars = self.catInfo.numCols-len(self.catInfo.categories)
        
        i = self.catInfo.numVars
        self.catInfo.featureToCol = list(range(i))
        for cat in self.catInfo.categories:
            self.catInfo.featureToCol.extend([i]*len(cat["labels"]))
            i += 1

        # Number of variables+category sizes
        self.catInfo.numFeatures = len(self.catInfo.featureToCol)
        # Number of input neurons (features+masks)
        self.catInfo.numInputs = self.catInfo.numFeatures+self.catInfo.numCols

        # Array of inputs
        self.data = dfNums.to_numpy()
        # Current position in data
        self.dataPos = 0 
        self.numRows = self.data.shape[0]
        
    def getTrainingData(self, batchSize=100, p=0.1):
        dataEnd = self.dataPos+batchSize
        if dataEnd <= self.numRows:
            batch = self.data[self.dataPos:dataEnd]
        else:
            dataEnd -= self.numRows
            batch = np.concatenate((self.data[self.dataPos:],
                                    self.data[:dataEnd]),
                                   axis = 0)
        self.dataPos = dataEnd
        mask = np.random.random((batchSize, self.catInfo.numCols)) < p
        maskedData = np.concatenate((batch, mask), axis = 1)
        #print(mask)
        for f in range(self.catInfo.numFeatures):
            c = self.catInfo.featureToCol[f]
            maskedData[:, f] *= mask[:, c]
            #print(data)
        return maskedData, batch

#batch = dfNumpy[:10]
#print(featureToCol)
#print(batch)
#maskedBatch = maskData(batch)
#print(maskedBatch)

class Imputer(nn.Module):
    def __init__(self, csvData):

        self.catInfo = csvData.catInfo

        self.hidden1 = 20
        self.hidden2 = 10
        
        self.feedforward = nn.Sequential(
            nn.Linear(self.catInfo.numInputs, hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU()
        )

        self.varPredict = nn.Linear(self.hidden2, self.catInfo.numVars)
        self.varLoss = nn.MSELoss(reduction="sum")
        
        self.catPredicts = []
        self.catLosses = []
        for cat in self.catInfo.categories:
            catLen = len(cat["labels"])
            self.catPredicts.append(nn.Linear(self.hidden2, catLen))
            self.catLosses.append(nn.LogSoftmax(dim=1))

    def forward(self, x):
        z = self.feedforward(x)
        self.y = self.varPredict(z)
        self.loss = self.varLoss(y)
        for i in range(len(self.catInfo.categories)):
            w = self.catPredicts[i](z)
            self.y = torch.cat(self.y, w)
            self.loss += self.catLosses(w)
        return self.y
        
csvData = CSVData("../data extraction/prepared_data.csv")
print(csvData.catInfo.categories)
