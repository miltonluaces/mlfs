# Dynamic Linear Models (DLM) with pydlm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import matplotlib.pyplot as plt
import pydlm


# Simple example (random walk)
n = 100
a = 1.0 + np.random.normal(0, 5, n) # the intercept
x = np.random.normal(0, 2, n) # the control variable
b = 3.0 # the coefficient
y = a + b * x

dlm = pydlm.dlm(y)
dlm = dlm + pydlm.trend(degree=0, discount=0.98, name='a', w=10.0)
dlm = dlm + pydlm.dynamic(features=[[v] for v in x], discount=1, name='b', w=10.0)

# randomly generate data
data = [0] * 100 + [3] * 100

# creadte model
dlm = pydlm.dlm(data)

# add components
dlm = dlm + pydlm.trend(1, name='lineTrend', w=1.0) # covariance=1
dlm = dlm + pydlm.seasonality(7, name='7day', w=1.0)
dlm = dlm + pydlm.autoReg(degree=3, data=data, name='ar3', w=1.0)
dlm.ls()

# delete unwanted component
dlm.delete('7day')
dlm.ls()

# Analize results
dlm.fitForwardFilter()
dlm.fitBackwardSmoother()

# Plot
dlm.plot()
dlm.turnOff('smoothed plot')
dlm.plot()
dlm.turnOff('multiple plots')
dlm.plot()

# Re-fit
dlm = dlm + pydlm.seasonality(4)
dlm.ls()
dlm.fit()

# Missing observations
data = [1, 0, 0, 1, 0, 0, None, 0, 1, None, None, 0, 0]
dlm = pydlm.dlm(data) + pydlm.trend(1, w=1.0)
dlm.fit()  # fit() will fit both forward filter and backward smoother

# Tunning
tuner = pydlm.modelTuner(method='gradient_descent', loss='mse')
dlm2 = tuner.tune(dlm, maxit=100)
dlm2.fit()

print("MSE 1 : ", dlm.getMSE())
print("MSE 2 : ", dlm2.getMSE())

# Filtered results
filtMean = dlm.getMean(filterType='forwardFilter')
smoothMean = dlm.getMean(filterType='backwardSmoother')
filtVar = dlm.getVar(filterType='forwardFilter')
smoothVar = dlm.getVar(filterType='backwardSmoother')

plt.plot(filtMean)
plt.plot(smoothMean)
plt.plot(filtVar)
plt.plot(smoothVar)
plt.show()

filtCI = dlm.getInterval(filterType='forwardFilter')
smoothCI = dlm.getInterval(filterType='backwardSmoother')

# get the residual time series
#res = dlm.getResidual(filterType='backwardSmoother')

# get the filtered and smoothed mean for a given component
#filtTrend = dlm.getMean(filterType='forwardFilter', name='lineTrend')
#smoothTrend = dlm.getMean(filterType='backwardSmoother', name='lineTrend')

# get the latent states
#allStates = dlm.getLatentState(filterType='forwardFilter')
#trendStates = dlm.getLatentState(filterType='forwardFilter', name='lineTrend')

# Update
#dlm = dlm([]) + pydlm.trend(1) + pydlm.seasonality(7)
#for t in range(0, len(data)):
    #dlm.append([data[t]])
    #dlm.fitForwardFilter()
#filteredObs = dlm.getFilteredObs()
