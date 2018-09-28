import numpy as np
import scipy.linalg as scl
from sklearn.preprocessing import PolynomialFeatures
from Franke import FrankeFunction
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from types import  MethodType
from tqdm import tqdm
import sys
import matplotlib.pylab as plt


def check_types(*args):
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
    @check_types(np.ndarray, np.ndarray, np.ndarray, int)
    def __init__(self, x, y, z, deg):
        """
        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :param deg: The degree of polynomial we try to fit the data
        :type XY: array
        :type z: array
        :type deg: int
        """

        self.x = x
        self.y = y
        self.z = z
        self.deg = deg
        self.N = x.shape[0]
        self.lamb = 0.1

        xy = np.append(x, y, axis=1)
        poly = PolynomialFeatures(degree = deg)
        self.XY = poly.fit_transform(xy)
    """
    @property
    def set_ols(self):
        self.regressionmethod = getattr(self, 'ols')
    @property
    def set_ridge(self):
        self.regressionmethod = getattr(self, 'ridge')
    @property
    def set_lasso(self):
        self.regressionmethod = getattr(self, 'lasso')
    """

    def split_data(self, folds = None, frac = None, shuffle = False):

        if folds != None and frac != None:
            print("Error: Both folds and frac given, give only one of them.")
            sys.exit(0)
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

        elif frac != None:
            nTest = int(np.floor(frac*XY.shape[0]))
            XY_Train = XY[:-nTest]
            XY_Test = XY[-nTest:]

            z_Train = z[:-nTest]
            z_Test = z[-nTest:]

            self.XY_Train = XY_Train
            self.XY_Test = XY_Test
            self.z_Train = z_Train
            self.z_Test = z_Test


    @check_types(np.ndarray, np.ndarray)
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
        self.var_ols = np.linalg.pinv(XY.T @ XY)*varz
        return beta

    @check_types(np.ndarray, np.ndarray)
    def ridge(self, XY = None, z = None):
        """
        Performes a Ridge regression linear fit

        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :param lamb: The regularization constant
        :type XY: array
        :type z: array
        :type lamb: float, int
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if XY is None: XY = self.XY
        if z is None: z = self.z

        I = np.identity(XY.shape[1])
        #beta = scl.inv(XY.T @ XY + self.lamb*I) @ XY.T @ z

        U, s, Vt = scl.svd(XY, full_matrices=False)
        d = (s/(s **2 + self.lamb)).reshape(XY.shape[1], 1)
        beta = Vt.T @ (d * U.T @ z)

        return beta


    @check_types(np.ndarray, np.ndarray)
    def lasso(self, XY = None, z = None):
        """
        Performes a Lasso regression linear fit

        :param XY: A matrix of polynomialvalues
        :param z: The values we are trying to fit
        :param lamb: The regularization constant
        :type XY: array
        :type z: array
        :type lamb: float, int
        :return: The coefficient of the fitted polynomial
        :rtype: array
        """

        if XY is None: XY = self.XY
        if z is None: z = self.z

        lass = Lasso([float(self.lamb)], fit_intercept=False)
        lass.fit(XY, z)

        beta = (lass.coef_).reshape(XY.shape[1], 1)

        return beta

    @check_types(np.ndarray, np.ndarray)
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


    @check_types(np.ndarray, np.ndarray)
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
        I dont fucking know
        """
        nTrain = self.XY_Train.shape[0]
        nTest = self.XY_Test.shape[0]

        zpreds = np.zeros((nBoots, nTest))

        for i in tqdm(range(nBoots)):
            idx = np.random.choice(nTrain, nTrain)
            XY_boot = self.XY_Train[idx]
            z_boot = self.z_Train[idx]

            beta = regressionmethod(XY_boot, z_boot)
            zpredict = self.XY_Test @ beta
            zpreds[i] = zpredict.flatten()

        z_avg = np.average(zpreds, axis = 0).reshape(-1, 1)
        z_var = np.var(zpreds, axis = 0).reshape(-1, 1)

        return z_avg, z_var

    @check_types(int, MethodType)
    def kfold(self, nfolds, regressionmethod):
        """
        I dont fucking know
        """


        XY_folds = np.array_split(self.XY, nfolds, axis = 0)
        z_folds = np.array_split(self.z, nfolds, axis = 0)
        mse_ave = 0

        for i in range(nfolds):

            XY_Train = XY_folds.copy()
            XY_Test = XY_Train.pop(i)
            XY_Train = np.concatenate(XY_Train)

            Z_Train = z_folds.copy()
            Z_Test = Z_Train.pop(i)
            Z_Train = np.concatenate(Z_Train)

            beta = regressionmethod(XY_Train, Z_Train)
            zpredict = XY_Test @ beta
            mse = self.MSE(Z_Test, zpredict)
            mse_ave += mse

        mse_ave /= nfolds
        print("k-fold average MSE: ", mse_ave)

    #def statistics():



    @check_types(int, int, int)
    def testfunc(self, a, b, c):
        pass
