from math import sqrt
from random import randrange
from random import seed


# Calculate the Euclidean distance between two vectors
def Dist(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1): distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the best matching unit
def GetWinner(values, vec):
	dists = list()
	for v in values:
		dist = Dist(v, vec)
		dists.append((v, dist))
	dists.sort(key=lambda tup: tup[1])
	return dists[0][0]

# Create a random sample
def GetRandomSample(train):
	nRows = len(train)
	nFeats = len(train[0])
	sample = [train[randrange(nRows)][i] for i in range(nFeats)]
	return sample

# Train a set of vectors
def Train(train, nVectors, lrate, epochs):
    
	values = [GetRandomSample(train) for i in range(nVectors)]
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs))) # lrate decay
		totError = 0.0
		for vec in train:
			bmu = GetWinner(values, vec)
			for i in range(len(vec)-1):
				error = vec[i] - bmu[i]
				totError += error**2
				if bmu[-1] == vec[-1]: bmu[i] += rate * error # takes nothing 
				else: bmu[i] -= rate * error   # takes all
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, totError))
	return values

# Testing
dataset = [[2.7810836,2.550537003,0],[1.465489372,2.362125076,0],[3.396561688,4.400293529,0],[1.38807019,1.850220317,0],[3.06407232,3.005305973,0],[7.627531214,2.759262235,1],[5.332441248,2.088626775,1],[6.922596716,1.77106367,1],[8.675418651,-0.242068655,1],[7.673756466,3.508563011,1]]
lrate = 0.3
epochs = 10
nVectors = 2
values = Train(dataset, nVectors, lrate, epochs)
print('Values: ', values)
