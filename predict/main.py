import numpy as np
import pandas as pd
import torch
from torch import nn, optim, Tensor
import matplotlib.pyplot as plt

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

        #print(dfNums.columns.values)
        
        # Array of inputs
        self.data = dfNums.to_numpy()
        # Current position in data
        self.numRows = self.data.shape[0]
        
    def getTrainingData(self, numIters, batchSize):
        dataPos = 0
        for i in range(numIters):
            #p = 1-(i/numIters)
            #p *= p
            p = 0.1
            dataEnd = dataPos+batchSize
            if dataEnd <= self.numRows:
                batch = self.data[dataPos:dataEnd]
            else:
                dataEnd -= self.numRows
                batch = np.concatenate((self.data[dataPos:],
                                        self.data[:dataEnd]),
                                       axis = 0)
            dataPos = dataEnd
            mask = np.random.random((batchSize, self.catInfo.numCols)) < p
            maskedData = np.concatenate((batch, mask), axis = 1)
            #print(mask)
            for f in range(self.catInfo.numFeatures):
                c = self.catInfo.featureToCol[f]
                maskedData[:, f] *= mask[:, c]
                #print(data)
            yield Tensor(maskedData), Tensor(batch)

#batch = dfNumpy[:10]
#print(featureToCol)
#print(batch)
#maskedBatch = maskData(batch)
#print(maskedBatch)

class Imputer(nn.Module):
    def __init__(self, csvData):
        super().__init__()
        self.catInfo = csvData.catInfo

        self.hidden1 = 20
        self.hidden2 = 10
        
        self.feedforward = nn.Sequential(
            nn.Linear(self.catInfo.numInputs, self.hidden1),
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
            self.catPredicts.append(nn.Sequential(
                nn.Linear(self.hidden2, catLen),
                nn.LogSoftmax(dim=1)
            ))
            self.catLosses.append(nn.NLLLoss(reduction="sum"))

    def forward(self, x, y):
        z = self.feedforward(x)
        #print(self.catInfo.numVars)
        self.yHat = self.varPredict(z)
        j = self.catInfo.numVars
        self.loss = self.varLoss(self.yHat, y[:, :j])
        for i in range(len(self.catInfo.categories)):
            w = self.catPredicts[i](z)
            self.yHat = torch.cat((self.yHat, w), 1)
            k = j+len(self.catInfo.categories[i]["labels"])
            self.loss += self.catLosses[i](w, torch.argmax(y[:, j:k], dim=1))
            j = k
        return self.yHat

csvData = CSVData("../data extraction/prepared_data.csv")
imputer = Imputer(csvData)

optimizer = optim.SGD(imputer.parameters(), lr=0.000001, momentum=0.9)

#print(csvData.catInfo.categories)

losses = []

for i, data in enumerate(csvData.getTrainingData(10000, 100)):
    # get the inputs; data is a list of [inputs, labels]
    x, y = data
    
    # zero the parameter gradients
    optimizer.zero_grad()
    
    # forward + backward + optimize
    yHat = imputer(x, y)
    loss = imputer.loss
    loss.backward()
    optimizer.step()
    
    # print statistics
    if i % 100 == 99:    # print every 2000 mini-batches
        print(f'{i + 1} loss: {loss}')
        losses.append(loss.detach().numpy())

PATH = './trained_model.pth'
torch.save(imputer.state_dict(), PATH)
plt.title("p=0.1 (heavy masking)")
plt.xlabel("Batch")
plt.ylabel("Combined loss")
plt.plot(losses)
plt.savefig("01.png")
