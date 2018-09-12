from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import numpy as np
from random import random, seed
from Franke import FrankeFunction

#np.set_printoptions(linewidth=np.nan)

N = 10
x = np.random.rand(N,1)
y = np.random.rand(N,1)

xy = np.append(x, y, axis=1)
z = FrankeFunction(x, y)

### SELF WRITTEN
#---------------------------------
deg = 5

XY = np.ones([N, 1])

count= 1
for i in range(1,deg+1):
    for j in range(i+1):
        if (i == 0) and (j==0):
            continue
        print("x^", i-j, "y^", j)
        col = x**(i-j)*y**j
        XY = np.append(XY, col, axis=1)
        count += 1
print("count: ", count)



beta = np.linalg.inv(XY.T.dot(XY)).dot(XY.T).dot(z)
zpredict = XY.dot(beta)

print("betas: ", beta.flatten())

#Mean squared error
mse = 1.0/N*np.sum((z - zpredict)**2)
print("Mean squared error: %.3f" % mse)
#R2-score
zmean = 1.0/N*np.sum(z)
r2 = 1 - np.sum((z - zpredict)**2)/np.sum((z - zmean)**2)
print('R2-score: %.3f' % r2_score(z, zpredict))


#xb = np.c_[np.ones((xy.shape[0])), ]

### SCIKIT FASIT
#----------------------------------

poly = PolynomialFeatures(degree=deg)
XY = poly.fit_transform(xy)

linreg = LinearRegression()
linreg.fit(XY,z)

print("beta 0: ", linreg.intercept_)
print("betas : ", linreg.coef_)

zpredict = linreg.predict(XY)

# The mean squared error
print("Mean squared error: %.3f" % mean_squared_error(z, zpredict))
# Explained variance score: 1 is perfect prediction
print('R2-score: %.3f' % r2_score(z, zpredict))
