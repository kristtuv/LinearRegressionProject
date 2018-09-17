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
import sys


def LeastSquares(N, deg, lamb, method='self'):


    x = np.random.rand(N,1)
    y = np.random.rand(N,1)
    xy = np.append(x, y, axis=1)

    if lamb != 0:
        noise = lamb*np.random.randn(N,1)
        z = FrankeFunction(x, y)*noise
    else:
        z = FrankeFunction(x,y)

    ### SELF WRITTEN
    #---------------------------------
    if method == 'self':

        XY = np.ones([N, 1])

        count= 1
        for i in range(1,deg+1):
            for j in range(i+1):
                if (i == 0) and (j==0):
                    continue
                #print("x^", i-j, "y^", j)
                col = x**(i-j)*y**j
                XY = np.append(XY, col, axis=1)
                count += 1
        #print("count: ", count)

        """ Alternative
        degree = 3 #Polynomial degree
        for deg in range(degree + 1):
            liste = np.arange(deg+1)
            for i, j in zip(liste, np.flip(liste, 0)):
                print(j, i)
        """

        beta = np.linalg.inv(XY.T.dot(XY)).dot(XY.T).dot(z)
        zpredict = XY.dot(beta)

        print("betas: (beta0, beta1, ...)\n", "-"*30, "\n", beta.flatten())

    ### SCIKIT
    #----------------------------------

    elif method == 'scikit':
        poly = PolynomialFeatures(degree=deg)
        XY = poly.fit_transform(xy)

        linreg = LinearRegression()
        linreg.fit(XY,z)

        print("beta 0: ", linreg.intercept_)
        print("betas : ", linreg.coef_)

        zpredict = linreg.predict(XY)


    else:
        print("Error: 'method' must be either 'self' for the self-written method, or 'scikit' to use\
                        the regression method of scikit learn.\n Exiting...")
        sys.exit(0)


    # STATISTICS
    #-------------------------------

    squared_error = np.sum((z - zpredict)**2)

    var_z = 1.0/(N - deg -1)*squared_error
    var_b = np.diag(np.linalg.inv(XY.T.dot(XY))*var_z)
    print("\nVariance betas: (Var(beta0), Var(beta1), ...)\n", "-"*50, "\n", var_b)

    #Mean squared error
    mse = 1.0/N*squared_error
    print("\nMean squared error: %.3f" % mse)
    #R2-score
    zmean = 1.0/N*np.sum(z)
    r2 = 1 - np.sum((z - zpredict)**2)/np.sum((z - zmean)**2)
    print('R2-score: %.3f' % r2)


# def Ridge(N, deg, lamb, method='self'):


if __name__=='__main__':
    LeastSquares(100, 3, 0)
