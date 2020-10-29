import numpy as np
from numpy import random


# Converts an integer to a reversed bitstring of specified size. E.g.: asBytes(3, 4) = [1, 1, 0, 0]; asBytes(3, 5) = [1, 1, 0, 0, 0]
def AsBytes(num, size):
    res = []
    for _ in range(size):
        res.append(num % 2)
        num //= 2
    return res

# Generate example addition. a (1st term), b (2nd term), c (addition a+b). All represented as reversed strings
def GenExample(nBits):
    a = random.randint(0, 2**(nBits - 1) - 1)
    b = random.randint(0, 2**(nBits - 1) - 1)
    res = a + b
    return (AsBytes(a,  nBits),
            AsBytes(b,  nBits),
            AsBytes(res,nBits))

# Generate batchSize instances of the addition problem. x: 2 numbers to be added, y: addition result (all b: bit index from the end, i: idx, n [0,1] for terms, 0 for result.
def GenBatch(nBits, batchSize):
    x = np.empty((batchSize, nBits, 2))
    y = np.empty((batchSize, nBits, 1))

    for i in range(batchSize):
        a, b, r = GenExample(nBits)
        x[i, :, 0] = a
        x[i, :, 1] = b
        y[i, :, 0] = r
    return x, y

