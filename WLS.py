# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 01:19:28 2020

@author: Ratan Singh
"""

import pandas as pd
import numpy as np
from OLS import computeGradient, computeL2Cost, normalize, denormalize, gradientDescent
from matplotlib import pyplot as plt
import matplotlib

dataset = pd.read_excel('CA_Test.xlsx')

# Fitting an OLS
n = dataset.shape[1] - 1
X = np.array(dataset.iloc[:,0:n])
y = np.array(dataset.iloc[:,n])

# Normalizing the features

X,mu,sigma = normalize(X)


# Adding ones for bias
X = np.hstack((X, np.ones((X.shape[0],1))))


# Defining parameters for gradient descent
initalWeights = np.random.random([1,X.shape[1]])
maxIter = 1000
learningRate = 0.1


# Training a OLS Regression
weights = initalWeights
cost = []

for i in range(maxIter):
	yp = np.matmul(X, weights.T)
	J = computeL2Cost(y, yp)
	G = computeGradient(X, y, yp)
	weights = gradientDescent(weights, G, learningRate)
	if i%10 ==0:
		print("Cost of the model is {}".format(J))
	cost.append(J)


print("Weights after the training are :: {}".format(weights))


# Plotting the training loss curve
plt.plot(range(0,len(cost)),cost)
plt.title('Cost per iterations')
plt.show()


# Prediction using the model
yp = np.matmul(X, weights.T).ravel()

#%%

# Computing the residuals
epsilon = (yp - y)


# You can choose either absolute or squared value of residual to fit
squared_residuals = epsilon**2
absolute_residuals = abs(epsilon)
r = squared_residuals


# Fit OLS with predictors to above squared/absolute residuals
residual_initialWeights = np.random.random([1,X.shape[1]])
residual_maxIter = 1000
learningRate = 0.1


# Training a OLS Regression on Residual vs Predictors
residual_weights = residual_initialWeights

for i in range(residual_maxIter):
	rp = np.matmul(X, residual_weights.T)
	G = computeGradient(X, r, rp)
	residual_weights = gradientDescent(residual_weights, G, learningRate)

print("Residuals Weights after the training are :: {}".format(residual_weights))


# Variances will be predicted residuals with OLS
variances = np.matmul(X, residual_weights.T)

#%%
# Weights are reciprocal of this or reciprocal of square root of this (Depends from sources to sources)
regression_weights = 1.0/np.sqrt(variances)
regression_weights_matrix = np.diagflat(np.array(regression_weights).ravel())


X_weighted = np.dot(regression_weights_matrix, X)
Y_weighted = np.dot(regression_weights_matrix, y.reshape([-1,1]))


#%%

# Fit OLS with predictors to above squared/absolute residuals
WLS_initialWeights = np.random.random([1,X.shape[1]])
WLS_maxIter = 1000
learningRate = 0.1


# Training a OLS Regression on Residual vs Predictors
WLS_weights = WLS_initialWeights

for i in range(WLS_maxIter):
	yp_weighted = np.matmul(X_weighted, WLS_weights.T)
	G = computeGradient(X_weighted, Y_weighted, yp_weighted)
	WLS_weights = gradientDescent(WLS_weights, G, learningRate)

print("Weighted Regression Coefficients after the training are :: {}".format(WLS_weights))



 