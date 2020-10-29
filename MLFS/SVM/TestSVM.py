import numpy as np
import SVM as svm


data = {-1:np.array([[1,7],[2,8],[3,8],]), 1:np.array([[5,1],[6,-1],[7,3],])}


w, b, minVal, maxVal = svm.Train(data=data)

test = [[0,10],[1,3],[3,4],[3,5],[5,5],[5,6],[6,-5],[5,8]]

Cl = []
for t in test:
    cl = svm.Predict(t, w, b)
    Cl.append(cl)

svm.Show(data, test, Cl, w, b, minVal, maxVal)
