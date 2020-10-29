from scipy import *
from scipy.linalg import norm, pinv 

class RBF:
     
    # Constructor
    def __init__(self, indim, numCenters, outdim):
        self.indim = indim
        self.outdim = outdim
        self.numCenters = numCenters
        self.centers = [random.uniform(-1, 1, indim) for i in range(numCenters)]
        self.beta = 8
        self.W = random.random((self.numCenters, self.outdim))
         
    # RBF
    def basisFun(self, c, d):
        assert len(d) == self.indim
        return exp(-self.beta * norm(c-d)**2)
    
    
         
    # Calculate activations of RBFs
    def calcAct(self, X):
        G = zeros((X.shape[0], self.numCenters), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi,ci] = self.basisFun(c, x)
        return G
    
    # X: matrix of dimensions n x indim , y: column vector of dimension n x 1. Originally random center vectors.
    def Train(self, X, Y):
        rnd_idx = random.permutation(X.shape[0])[:self.numCenters]
        self.centers = [X[i,:] for i in rnd_idx]
        print("center", self.centers)
        G = self.calcAct(X)
        print(G)
         
        self.W = dot(pinv(G), Y)

    # X: matrix of dimensions n x indim  
    def Test(self, X):
        G = self.calcAct(X)
        Y = dot(G, self.W)
        return Y
 
   