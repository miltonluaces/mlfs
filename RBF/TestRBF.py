from scipy import *
from matplotlib import pyplot as plt
from RBF import RBF

# Generate data (set y and random noise)
n = 100
     
x = mgrid[-1:1:complex(0,n)].reshape(n, 1)
y = sin(3*(x+0.5)**3 - 1)
     
# RBF regression
rbf = RBF(1, 10, 1)
rbf.Train(x, y)
z = rbf.Test(x)
       
# Plot original data and learned model
plt.figure(figsize=(12, 8))
plt.plot(x, y, 'k-')
plt.plot(x, z, 'r-', linewidth=2)
     
# Plot rbfs (each RF prediction lines)
plt.plot(rbf.centers, zeros(rbf.numCenters), 'gs')     
for c in rbf.centers:
    cx = arange(c-0.7, c+0.7, 0.01)
    cy = [rbf.basisFun(array([cx_]), array([c])) for cx_ in cx]
    plt.plot(cx, cy, '-', color='gray', linewidth=0.2)
     
plt.xlim(-1.2, 1.2)
plt.show()

