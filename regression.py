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
x = np.random.rand(100,1)
y = np.random.rand(100,1)

xy = np.append(x, y, axis=1)
z = FrankeFunction(x, y)

poly = PolynomialFeatures(degree=3)
XY = poly.fit_transform(xy)

### SCIKIT FASIT
linreg = LinearRegression()
linreg.fit(XY,z)

print("beta 0: ", linreg.intercept_)
print("betas : ", linreg.coef_)

zpredict = linreg.predict(XY)

#print(z)
#print(zpredict)


# The mean squared error
print("Mean squared error: %.3f" % mean_squared_error(z, zpredict))
# Explained variance score: 1 is perfect prediction
print('R2-score: %.3f' % r2_score(z, zpredict))
