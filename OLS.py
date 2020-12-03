# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd


"""
Gradient descent algorithm can be given as
weight_new = weigh_old - learning_rate*gradient
"""	
def gradientDescent(weight, gradient, alpha=0.1):
    weight = weight - alpha*gradient
    return weight


"""
Gradient can be computed by chain rule of differentiation.
on differentiating the L2 cost function with respect to w
y  :  True target value
yp :  Predicted target value
X  :  Features
"""
def computeGradient(X,y,yp):
    X = np.array(X)
    y = np.array(y).reshape([-1,1])
    yp = np.array(yp).reshape([-1,1])
    
    m = y.shape[0]
    
    gradient = np.matmul((yp-y).T,X)/m
    return gradient



"""
  L2 cost can be given as 
  J = mean[(yp-y)^2] / 2
  J  :  Cost funtion
  y  :  True target value
  yp :  Predicted target value
"""
def computeL2Cost(y,yp):
    y = np.array(y).ravel()
    yp = np.array(yp).ravel()
    J = np.mean((yp-y)**2)/2
    return J


"""
X 		:  Denormalized Feature matrix (m,n)
Returns 
mu  	:  mean array (n,)
sigma	:  Standard deviation array (n,)
X_normalized = X - mu / sigma
"""
def normalize(X):
    X = np.array(X)    
    mu = np.mean(X,axis = 0)
    sigma = np.std(X,axis = 0)
    X_normalized = (X-mu)/sigma
    return X_normalized, mu, sigma



"""
X 		:  Normalized Feature matrix (m,n)
mu  	:  mean array (n,)
sigma	:  Standard deviation array (n,)
Returns denormalized matrix X (m,n)
"""
def denormalize(X,mu,sigma):    
    X = np.array(X)
    X = X*sigma+mu
    return X
