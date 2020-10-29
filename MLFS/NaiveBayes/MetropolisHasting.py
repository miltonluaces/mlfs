import numpy as np
import math
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import time


def Metropolis(q, n):
    r = np.zeros(1)
    p = q(r[0])
    P = []
    
    for i in range(n):
        ri = r + np.random.uniform(-1,1)
        pi = q(ri[0])
        if pi >= p:
            p = pi
            r = ri
        else:
            u = np.random.rand()
            if u < pi/p:
                p = pi
                r = ri
        P.append(r)
    
    return np.array(P)

    


# Testing

# Parameters
mu = 0
s = 1
n =10000
                 
# Function : 1/(√(2πs2) . e^(x-m)^2/2s2)
def q(x):
    return (1/(math.sqrt(2*math.pi*s**2)))*(math.e**(-((x-mu)**2)/(2*s**2)))

P = Metropolis(q, n)

# Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1,)
ax.hist(P, bins=1000)    
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()