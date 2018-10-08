"""
Nasty program for demonstrating RSS
"""
import sys
sys.path.append('../../')
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotparams

x = np.random.randn(10).reshape(10, 1)
y = 2*x + np.random.randn(10).reshape(10,1)

poly = PolynomialFeatures(degree = 1)
X = poly.fit_transform(x)
linreg = LinearRegression()
linreg.fit(X, y)
ypredict = linreg.predict(X)
e = y - ypredict


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.plot(x, y, 'o', label='TrainingData', color='green')
ax.errorbar(x, ypredict, yerr=[0*e,e], label='Fitted line with error', color='yellow', ecolor='red')
ax.legend()
plt.tight_layout()
# plt.savefig('plots/OLS_RSS_example.png')
plt.show()

