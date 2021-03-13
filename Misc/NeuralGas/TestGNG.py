import numpy as np
from GNG import GrowingNeuralGas
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os
import shutil

if os.path.exists('visualization/sequence'): shutil.rmtree('visualization/sequence')
os.makedirs('visualization/sequence')

n_samples = 2000
dataset_type = 'moons'
data = None

# Load data
print('Preparing data...')
if dataset_type == 'blobs': data = datasets.make_blobs(n_samples=n_samples, random_state=8)
elif dataset_type == 'moons': data = datasets.make_moons(n_samples=n_samples, noise=.05)
elif dataset_type == 'circles': data = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
data = StandardScaler().fit_transform(data[0])
print('Done.')

# Fit model
print('Fitting neural network...')
gng = GrowingNeuralGas(data)
gng.fit_network(e_b=0.1, e_n=0.006, a_max=10, l=200, a=0.5, d=0.995, passes=8, plot_evolution=True)
print('Found %d clusters.' % gng.number_of_clusters())

# Plot results
gng.plot_clusters(gng.cluster_data())
