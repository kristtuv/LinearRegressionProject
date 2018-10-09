"""
Not in use anymore
"""
import sys
sys.path.append('../')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import numpy as np
from random import random, seed
from Franke import FrankeFunction
import sys
import scipy as sc

x = np.random.rand(10,1)
y = np.random.rand(10,1)
noise = 0.5*np.random.randn(10,1)

z = FrankeFunction(x, y)

xy = np.append(x, y, axis=1)
poly = PolynomialFeatures(degree=2)
XY = poly.fit_transform(xy)

def kfold(XY, z, k):

    XY_folds = np.array_split(XY, k, axis = 0)
    z_folds = np.array_split(z, k, axis = 0)

    print(XY_folds, "\n\n")

    #print(np.concatenate([z_folds[0], z_folds[1]]))
    for i in range(k):

        #ZTrain = z_folds.copy()
        #ZTest = ZTrain.pop(i)
        #print(np.concatenate(ZTrain))
        #print(ZTest, "\n")

        XYTrain = XY_folds.copy()
        XYTest = XYTrain.pop(i)
        print(np.concatenate(XYTrain))
        print(XYTest, "\n")



kfold(XY, z, 6)

"""
if sigma != 0:
    #noise = sigma*np.random.randn(N,1)
    z = FrankeFunction(x, y)*noise
else:
    z = FrankeFunction(x,y)
"""

"""
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
"""

#def Regression(N, deg, sigma, lamb, method='ls', stats = True):
def Regression(XY, z, lamb, method):
    """ method = 'ls', 'ridge'"""

    if method == 'ls':

        beta = sc.linalg.inv(XY.T.dot(XY)).dot(XY.T).dot(z)


    elif method == 'ridge':

        I = np.identity(XY.shape[1])

        beta = np.linalg.inv(XY.T.dot(XY) + lamb*I).dot(XY.T).dot(z)

    else:
        print("Error: 'method' must either be 'ls' or 'ridge'. \nExiting... ")
        sys.exit(0)

    print("betas: (beta0, beta1, ...)\n", "-"*30, "\n", beta.flatten())
    #if stats == True:
    #    statistics(XY, z, zpredict, deg, lamb, method)

    return beta

def predict(XY, beta):

    zpredict = XY.dot(beta)

    return zpredict



def Regression_scikit(N, deg, sigma, lamb, method='ls', stats = True):

    #poly = PolynomialFeatures(degree=deg)
    #XY = poly.fit_transform(xy)

    if method == 'ls':

        linreg = LinearRegression(fit_intercept = False)
        linreg.fit(XY,z)

        print("beta 0: ", linreg.intercept_)
        print("betas : ", linreg.coef_)

        zpredict = linreg.predict(XY)

    elif method == 'ridge':

        ridge=RidgeCV([float(lamb)])
        ridge.fit(XY,z)
        zpredict = ridge.predict(XY)
        print("beta 0: ", ridge.intercept_)
        print("betas : ", ridge.coef_)

    elif method == 'lasso':

        lasso = Lasso([float(lamb)])
        lasso.fit(XY,z)
        ypredict = lasso.predict(XY)
        print("beta 0: ", lasso.intercept_)
        print("betas : ", lasso.coef_)


    else:
        print("Error: 'method' must be either 'ls', 'ridge', or 'lasso'. \nExiting...")
        sys.exit(0)

    if stats == True:
        statistics(XY, z, zpredict, deg, lamb, method)


def statistics(XY, z, zpredict, deg, lamb, method):

    print("Statistic for the %s method:" % (method))
    print("="*30)

    N = XY.shape[0]

    squared_error = np.sum((z - zpredict)**2)
    var_z = 1.0/(N - deg -1)*squared_error

    if method == 'ls':

        var_b = np.diag(np.linalg.inv(XY.T.dot(XY))*var_z)
        print("\nVariance betas: (Var(beta0), Var(beta1), ...)\n", "-"*50, "\n", var_b)

    elif method == 'ridge':

        # Maybe eaiser with SVD

        I = np.identity(XY.shape[1])

        XY2 = XY.T.dot(XY)
        W = np.linalg.inv(XY2 + lamb*I).dot(XY2)

        var_b = np.diag(W.dot(np.linalg.inv(XY2)).dot(W.T))*var_z
        print("\nVariance betas: (Var(beta0), Var(beta1), ...)\n", "-"*50, "\n", var_b)

    #Mean squared error
    mse = 1.0/N*squared_error
    print("\nMean squared error: %.3f" % mse)
    #R2-score
    zmean = 1.0/N*np.sum(z)
    r2 = 1 - np.sum((z - zpredict)**2)/np.sum((z - zmean)**2)
    print('R2-score: %.3f' % r2)

    print("="*30)


#beta1 = Regression(XY, z, 0, 'ls')
#Regression_scikit(10, 3, 0, 0, method='ls', stats = False)



#Regression(100, 3, 0, 0.01, 'ridge', stats = False)
#Regression(100, 3, 0.5, 2.0, 'ridge', stats = False)
#Regression_scikit(100, 3, 0, 0.01, method='ridge')
#Regression_scikit(100, 3, 0, 0.01, method='lasso')
