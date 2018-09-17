from franke import Franke
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import sys


class OLSsk:
    def __init__(self, Xtrain, Ytrain, Function, degree):
        self.x = Xtrain
        self.y = Ytrain
        self.f = Function
        self.degree = degree
        pass
    def compute(self):
        xy = np.append(self.x, self.y, axis=1)
        poly = PolynomialFeatures(self.degree)
        XY = poly.fit_transform(xy)
        linreg = LinearRegression()
        linreg.fit(XY,self.f)
        predicted = linreg.predict(XY)
        return linreg, predicted

    def prettyprint(self, precision=3):
        self.precision=precision
        np.set_printoptions(linewidth=150, precision=self.precision)
        linreg, predicted = self.compute()
        beta = np.append(linreg.intercept_, linreg.coef_)
        printbeta = 'Beta: '+str(beta) 

        msg = "Printing values for Ordinary Least Squares"
        print('#'*len(printbeta))
        print(msg)
        # print('#'*len(printbeta))
        print(printbeta)
        print("Mean squared error: %.3f" % mean_squared_error(self.f, predicted))
        # # Explained variance score: 1 is perfect prediction
        print('R2-score: %.3f' % r2_score(self.f, predicted))
        print('#'*len(printbeta))

class Ridgesk:
    def __init__(self, Xtrain, Ytrain, Function, degree):
        self.x = Xtrain
        self.y = Ytrain
        self.f = Function
        self.degree = degree

    def compute(self):
        xy = np.append(self.x, self.y, axis=1)
        poly = PolynomialFeatures(self.degree)
        XY = poly.fit_transform(xy)
        linreg = Ridge() 
        linreg.fit(XY,self.f)
        predicted = linreg.predict(XY)
        return linreg, predicted

    def prettyprint(self, precision=3):
        self.precision=precision
        np.set_printoptions(linewidth=150, precision=self.precision)
        linreg, predicted = self.compute()
        beta = np.append(linreg.intercept_, linreg.coef_)
        printbeta = 'Beta: '+str(beta) 

        msg = "Printing values for Ridge regression"
        print('#'*len(printbeta))
        print(msg)
        # print('#'*len(printbeta))
        print(printbeta)
        print("Mean squared error: %.3f" % mean_squared_error(self.f, predicted))
        # # Explained variance score: 1 is perfect prediction
        print('R2-score: %.3f' % r2_score(self.f, predicted))
        print('#'*len(printbeta))


class Lassosk:
    def __init__(self, Xtrain, Ytrain, Function, degree):
        self.x = Xtrain
        self.y = Ytrain
        self.f = Function
        self.degree = degree

    def compute(self):
        xy = np.append(self.x, self.y, axis=1)
        poly = PolynomialFeatures(self.degree)
        XY = poly.fit_transform(xy)
        linreg = Lasso() 
        linreg.fit(XY,self.f)
        predicted = linreg.predict(XY)
        return linreg, predicted

    def prettyprint(self, precision=3):
        self.precision=precision
        np.set_printoptions(linewidth=150, precision=self.precision)
        linreg, predicted = self.compute()
        beta = np.append(linreg.intercept_, linreg.coef_)
        printbeta = 'Beta: '+str(beta) 

        msg = "Printing values for Lasso regression"
        print('#'*len(printbeta))
        print(msg)
        # print('#'*len(printbeta))
        print(printbeta)
        print("Mean squared error: %.3f" % mean_squared_error(self.f, predicted))
        # # Explained variance score: 1 is perfect prediction
        print('R2-score: %.3f' % r2_score(self.f, predicted))
        print('#'*len(printbeta))


