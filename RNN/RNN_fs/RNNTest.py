import numpy as np
from Utils.Admin.Standard import *
from Preprocess import GetSentenceData
from Rnn import RNN

wordDim = 1000
nHid = 100
XTrain, yTrain = GetSentenceData(csvPath + 'reddit-comments.csv', wordDim)

np.random.seed(10)
rnn = RNN(wordDim, nHid)
losses = rnn.Train(XTrain[:100], yTrain[:100], m=0.005, epochs=5, evalLoss=1)