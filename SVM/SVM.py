import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from numpy.linalg import norm
style.use('ggplot')

def GetExtremes(data):
    values = np.asarray(list(data.values())).flatten()
    maxVal = max(values)
    minVal = min(values)
    return minVal, maxVal

# Training loop
def Train(data):
    minVal, maxVal = GetExtremes(data)
    steps = [maxVal * 0.1, maxVal * 0.01, maxVal * 0.001,] 

    # b Factors (we dont need to take as small of steps with b as we do w)
    bRangeFactor = 2 
    bFactor = 5
        
    # Support vectors : yi(xi.w+b) = 1
    options = {}
    transforms = [[1,1], [-1,1], [-1,-1], [1,-1]]
    best = maxVal * 10
    for step in steps:
        W = np.array([best,best]) # because its convex
        optimized = False
        while not optimized:
            for b in np.arange(-maxVal * bRangeFactor, maxVal * bRangeFactor, step * bFactor):
                for t in transforms:
                    foundOption = True
                    for yi in data:
                        for xi in data[yi]:
                            if not yi*(np.dot(W*t,xi) + b) >= 1:
                                foundOption = False  #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                    if foundOption: options[norm(W*t)] = [W*t,b]

            if W[0] < 0: optimized = True; print('step optimized')
            else: W = W - step

        norms = sorted([n for n in options])
        opt = options[norms[0]]
        W = opt[0]; b = opt[1] #||w|| : [w,b]
        best = opt[0][0] + step*2
            
    for yi in data:
        for xi in data[yi]:
            print(xi,' : ', yi * (np.dot(W,xi)+b)) 

    return(W,b,minVal,maxVal)

def Predict(features, w, b):
    return np.sign(np.dot(np.array(features),w)+b) # sign( x.w+b )

# Hyperplane equation : v = x.w+b
def GetHyperplane(x,w,b,v):
    return (-w[0]*x-b+v) / w[1]

def Show(data, features, classification, w, b, minVal, maxVal):
    colors = {1:'r',-1:'b'}
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in range(len(features)-1):
        fe = features[i]
        cl = classification[i]
        ax.scatter(fe[0], fe[1], s=200, marker='*', c=colors[cl])
     
    [[ax.scatter(x[0],x[1],s=100,color=colors[i]) for x in data[i]] for i in data]

    datarange = (minVal*0.9, maxVal*1.1)
    hypXmin = datarange[0]
    hypXmax = datarange[1]

    # positive support vector hyperplane : (w.x+b) = 1
    psv1 = GetHyperplane(hypXmin, w, b, 1)
    psv2 = GetHyperplane(hypXmax, w, b, 1)
    ax.plot([hypXmin,hypXmax],[psv1,psv2], 'k')

    # negative support vector hyperplane : (w.x+b) = -1
    nsv1 = GetHyperplane(hypXmin, w, b, -1)
    nsv2 = GetHyperplane(hypXmax, w, b, -1)
    ax.plot([hypXmin,hypXmax],[nsv1,nsv2], 'k')

    # positive support vector hyperplane : (w.x+b) = 0
    db1 = GetHyperplane(hypXmin, w, b, 0)
    db2 = GetHyperplane(hypXmax, w, b, 0)
    ax.plot([hypXmin,hypXmax],[db1,db2], 'y--')

    plt.show()
