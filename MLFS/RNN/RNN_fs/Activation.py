import numpy as np

class Sigmoid:

    def forward(self, x): 
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self, x, topDiff):
        output = self.forward(x)
        return (1.0 - output) * output * topDiff

class Tanh:

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, topDiff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * topDiff