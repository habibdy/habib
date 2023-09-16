# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 11:27:47 2023

@author: habib
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the asymmetric absolute loss function
def check(tau, x):

   return x * (tau - (x <= 0))
   
# Define the model function
def model(x, c):
    return  c[0] + c[1]*np.sin(x)

# Define the objective function to be minimized
def objective(c, tau, x, y):
    return np.sum(check(tau, y - model(x, c)))

# Generate toy data
np.random.seed(123)
x = np.linspace(0.0, 9.0, 1000)  # Avoids log(0) in the model
true_params = [4.0, 4.0]
noise = np.random.normal(0, 1, len(x))
y = model(x, true_params) + noise

# Fit quantile regression for different quantile levels
quantiles = np.array([0.25, 0.50, 0.55, 0.75, 0.95])
fit_params = []

for tau in quantiles:
    result = minimize(objective, x0=true_params, args=(tau, x, y), method='Nelder-Mead', options={'xtol': 1e-8, 'maxiter': 3000})
    fit_params.append(result.x)

# Create a new plot
plt.figure(figsize=(10, 7))

# Plot the original data points
plt.plot(x, y, 'ro', markersize=0.75 , color= 'black',label='Data')

# Plot each of the fit lines calculated
for i in range(quantiles.size):
    plt.plot(x, model(x, fit_params[i]), label=f"{100 * quantiles[i]:.0f}%")

# Format the plot
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Quantile Regression with Asymmetric Loss Function')
plt.show()
