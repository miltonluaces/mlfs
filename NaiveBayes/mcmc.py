# MCMC Gibbs Sampling from scratch
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def PxGivenY(y, mus, sigmas):
    mu = mus[0] + sigmas[1,0] / sigmas[0,0] * (y - mus[1])
    sigma = sigmas[0,0] - sigmas[1,0] / sigmas[1,1] * sigmas[1,0]
    return np.random.normal(mu,sigma)


def PyGivenX(x, mus, sigmas):
    mu = mus[1] + sigmas[0,1] / sigmas[1,1] * (x - mus[0])
    sigma = sigmas[1,1] - sigmas[0,1] / sigmas[0,0] * sigmas[0,1]
    return np.random.normal(mu,sigma)


def GibbsSampling(mus, sigmas, iter=10000):
    samples = np.zeros((iter, 2))
    y = np.random.rand() * 10

    for i in range(iter):
        x = PxGivenY(y, mus, sigmas)
        y = PyGivenX(x, mus, sigmas)
        samples[i,:] = [x,y]

    return samples




# Testing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

mus = np.array((5,5))
sigmas = np.array(((1,.9), (.9,1)))

samples = GibbsSampling(mus, sigmas)
sns.jointplot(samples[:,0], samples[:,1])
plt.show()
