import numpy as np
import scipy.linalg as scl
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from types import  MethodType
from tqdm import tqdm
import sys
import matplotlib.pylab as plt


def check_types(*args):
    """
    Decorator for checking types of function arguments
    Used during testing
    """
    def decorator(func):
        def wrapper(*argswrapper):
            argswrappercopy = (argswrapper[1:])
            for a, b in zip(args, argswrappercopy):
                if a is not type(b) and type(b) is not type(None) :
                    raise TypeError('See documentation for argument types')
            return func(*argswrapper)
        return wrapper
    return decorator



class LinReg:
    # @check_types(np.ndarray, np.ndarray, np.ndarray, int)
    def __init__(self, x, y, z, deg):
        """
        :param x: 1darray of x values
        :param y: 1darray of y values
        :param z: The values we are trying to fit
        :param deg: The degree of polynomial we try to fit the data
        :type x: ndarray
        :type y: ndarray
        :type z: ndarray
        :type deg: int
        """

        self.x = x
        self.y = y
        self.z = z
        self.deg = deg
        self.N = x.shape[0]
        self.lamb = 0.1

        nterms = np.sum(range(1, deg+2))
        self.XY = np.zeros((self.N, nterms))
        

        ####################
        #Calculating the design matrix
        ####################
        count = 0
        for i in range(deg+1):
            for j in range(i+1):

                self.XY[:,count] = (x**(i-j)*y**(j)).flatten()
                count += 1

        self.split_data(folds = 10, frac = 0.3)


    def split_data(self, folds = None, frac = None, shuffle = False):
        """
        Splits the data into training and test. Give either frac or folds

        param: folds: Number of folds
        param: frac: Fraction of data to be test data
        param: shuffle: If True: shuffles the design matrix 
        type: folds: int
        type: frac: float
        type: shuffle: Bool
        return: None
        """

        if folds == None and frac == None:
            print("Error: No split info received, give either no. folds or fraction.")
            sys.exit(0)

        XY = self.XY
        z = self.z

        if shuffle:
            XY = np.shuffle(XY, axis = 0)
            z = np.shuffle(z, axis = 0)

        if folds != None:
            XY_folds = np.array_split(XY, folds, axis = 0)
            z_folds = np.array_split(z, folds, axis = 0)

            self.XY_folds = XY_folds
            self.z_folds =  z_folds

        if frac != None:
            nTest = int(np.floor(frac*XY.shape[0]))
            XY_Train = XY[:-nTest]
            XY_Test = XY[-nTest:]

            z_Train = z[:-nTest]
            z_Test = z[-nTest:]

            self.XY_Train = XY_Train
            self.XY_Test = XY_Test
            self.z_Train = z_Train
            self.z_Test = z_Test


    # @check_types(np.ndarray, np.ndarray)
    def ols(self, XY = None, z = None):
        """
        Performes a Ordinary least squares linear fit

        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type XY: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if XY is None: XY = self.XY
        if z is None: z = self.z

        #beta = scl.inv(XY.T @ XY) @ XY.T @ z
        beta = np.linalg.pinv(XY) @ z

        zpredict = XY @ beta
        varz = 1.0/(XY.shape[0] - self.deg - 1)*np.sum((z - zpredict)**2)
        var = np.linalg.pinv(XY.T @ XY)*varz
        self.var_ols = var
        self.conf_ols = 1.96*np.sqrt(np.diag(var))
        return beta

    # @check_types(np.ndarray, np.ndarray)
    def ridge(self, XY = None, z = None):
        """
        Performes a Ridge regression linear fit

        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type XY: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if XY is None: XY = self.XY
        if z is None: z = self.z

        I = np.identity(XY.shape[1])
        XY_inv = scl.inv(XY.T @ XY + self.lamb*I)
        beta = XY_inv @ XY.T @ z
        zpredict = XY @ beta
        varz = 1.0/(XY.shape[0] - self.deg - 1)*np.sum((z - zpredict)**2)

        var = XY_inv @ XY.T @ XY @ XY_inv.T * varz
        self.var_ridge = var
        self.conf_ridge = 1.96*np.sqrt(np.diag(var))
        return beta


    # @check_types(np.ndarray, np.ndarray)
    def lasso(self, XY = None, z = None):
        """
        Performes a Lasso regression linear fit

        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :type XY: array
        :type z: array
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if XY is None: XY = self.XY
        if z is None: z = self.z

        lass = Lasso([float(self.lamb)], fit_intercept=False, max_iter = 5000)
        lass.fit(XY, z)

        beta = (lass.coef_).reshape(XY.shape[1], 1)

        return beta

    # @check_types(np.ndarray, np.ndarray)
    def MSE(self, z, zpred):
        """
        Finds the mean squared error of the real data and predicted values

        :param z: real data
        :param zpred: predicted data
        :type z: array
        :type zpred: array
        :return: The mean squared error
        :rtype: float
        """

        return 1.0/z.shape[0]*np.sum((z - zpred)**2)


    # @check_types(np.ndarray, np.ndarray)
    def R2(self, z, zpred):
        """
        Finds the R2 error of the real data and predicted values

        :param z: real data
        :param zpred: predicted data
        :type z: array
        :type zpred: array
        :return: The mean squared error
        :rtype: float
        """


        zmean = np.average(z)

        return 1 - np.sum((z - zpred)**2)/np.sum((z - zmean)**2)

    #@check_types(int, MethodType)
    def bootstrap(self, nBoots, regressionmethod):
        """
        Bootstraps the data defined in the instance
        and calculates the bias, variance, and average 
        mse training error and mse test error

        param: nBoots: Number of boostrap samples
        param: regressionmethod: ols, ridge or lasso
        type: nBoots: int
        type: regressionmethod: MethodType
        return: Bias, Variance, Training Error, Test Error
        rtype: float, float, float, float
        """
        nTrain = self.XY_Train.shape[0]
        nTest = self.XY_Test.shape[0]

        zpreds = np.zeros((nBoots, nTest))
        train_errors = np.zeros(nBoots)
        test_errors = np.zeros(nBoots)

        for i in tqdm(range(nBoots)):
            idx = np.random.choice(nTrain, nTrain)
            XY_boot = self.XY_Train[idx]
            z_boot = self.z_Train[idx]

            beta = regressionmethod(XY_boot, z_boot)
            zpred_train = XY_boot @ beta
            zpred_test = self.XY_Test @ beta

            zpreds[i] = zpred_test.flatten()
            train_errors[i] = self.MSE(z_boot, zpred_train)
            test_errors[i] = self.MSE(self.z_Test, zpred_test)

        train_error = np.average(train_errors)
        test_error = np.average(test_errors)

        z_avg = np.average(zpreds, axis = 0).reshape(-1, 1)
        variance = np.average(np.var(zpreds, axis = 0))
        bias = np.average((self.z_Test - z_avg)**2)

        return bias, variance, train_error, test_error

    # @check_types(int, MethodType)
    def kfold(self, nfolds, regressionmethod):
        """
        Split data into folds and run k-fold algorithm
        and calculate average r2 and mse error for 
        for training and test

        param: nfolds: number of folds
        type: nfolds: int
        param: regressionmethod: ols, ridge or lasso
        type: regressionmethod: MethodType
        return: mse_train, mse_test, r2_train, r2_test
        rtype: float, float, float, float
        """
        if nfolds != 10:
            self.split_data(folds = nfolds)

        XY_folds = self.XY_folds
        z_folds = self.z_folds

        mse_train = 0; mse_test = 0
        r2_train = 0; r2_test = 0

        for i in range(nfolds):

            XY_Train = XY_folds.copy()
            XY_Test = XY_Train.pop(i)
            XY_Train = np.concatenate(XY_Train)

            Z_Train = z_folds.copy()
            Z_Test = Z_Train.pop(i)
            Z_Train = np.concatenate(Z_Train)

            beta = regressionmethod(XY_Train, Z_Train)
            zpred_train = XY_Train @ beta
            zpred_test = XY_Test @ beta

            mse_train += self.MSE(Z_Train, zpred_train)
            mse_test += self.MSE(Z_Test, zpred_test)
            r2_train += self.R2(Z_Train, zpred_train)
            r2_test += self.R2(Z_Test, zpred_test)

        mse_train /= nfolds; mse_test /= nfolds
        r2_train /= nfolds; r2_test /= nfolds

        return mse_train, mse_test, r2_train, r2_test


