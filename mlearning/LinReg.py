#Version python 3.6
import numpy as np
import os
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[34m'
    YELLOW = '\033[93m'
    RED = '\033[31m'
    GREEN = '\033[32m'


class LinReg:
    def __init__(self, x, y, Function, Test):
        """ Trainingdata X and Y
            Function for fitting
            Fraction of TrainData set aside for testing
       """

        self.xTrain = x
        self.yTrain = y
        self.zTrain = Function
        self.N = len(self.xTrain)
        self.Test = Test
        self.PolyDegree()
        self._testshit()
        self.compute()


    def _testshit(self):
        nTestData = int (np.floor(self.N*self.Test)) #Number of training data
        TestDataIndex = np.random.choice(range(self.N), nTestData,\
                replace = False)
        #Storing TestData
        self.xTest = self.xTrain[TestDataIndex]
        self.yTest = self.yTrain[TestDataIndex]
        self.zTest = self.zTrain[TestDataIndex]

        #Removing TestData from TraingData
        self.xTrain = np.delete(self.xTrain, TestDataIndex, axis = 0 )
        self.yTrain =  np.delete(self.yTrain, TestDataIndex, axis = 0 )
        self.zTrain =  np.delete(self.zTrain, TestDataIndex, axis = 0 )

        #Rearranging Training data to match SckiKit and creating XY matrix
        self._rearrange()

    def bootstrap(self, nBoots = 1000):
        print('not implemented yet')
        exit()
        localXTrain = self.xTrain
        localYTrain = self.yTrain
        localZTrain = self.zTrain
        self.xTrain = np.zeros(nBoots)
        self.yTrain = np.zeros(nBoots)
        for k in range(0,nBoots):
            self.data[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
            self.xTrain[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
            self.yTrain[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
        # self.bootAvg = np.average(bootVec)
        # self.bootVar = np.var(bootVec)
        # self.bootStd = np.std(bootVec)
    def _rearrange(self):
        """
        Function for rearranging the order of the polynomial to match what is
        used by Scikit-learn. This is done for easy comparison of beta values.
        Will return a matrix of the form [1, x^ny^0, x^(n-1)y^1 ... x^0y^n],
        where n is the polynomial degree
        """
        N = len(self.xTrain)
        self.XY = np.ones([N, 1])
        for deg in range(1, self.degree + 1):
            liste = np.arange(deg+1)
            for i, j in zip(liste, np.flip(liste, 0)):
                col = self.xTrain**(j)*self.yTrain**i
                self.XY = np.append(self.XY, col, axis=1)

    def statistics(self):
        N = self.XY.shape[0]
        squared_error = np.sum((self.zTrain- self.zpredict)**2)
        zmean = 1.0/N*np.sum(self.zTrain)
        self.var_z = 1.0/(N - self.degree -1)*squared_error

        #Mean squared error
        self.mse = 1.0/N*squared_error
        #R2-score

        self.r2 = 1 - np.sum((self.zTrain - self.zpredict)**2)/np.sum((self.zTrain - zmean)**2)
    def __str__(self):
        rows, columns = os.popen('stty size', 'r').read().split()
        np.set_printoptions(precision=2, linewidth=int(columns))

        var_b = str(self.var_b)
        mse= str(self.mse)
        r2= str(self.r2)

        print(bcolors.YELLOW  + "="*min(len(var_b), int(columns)) + bcolors.ENDC)
        print(bcolors.UNDERLINE + bcolors.BLUE + "Variance of Beta:" + bcolors.ENDC)
        print(var_b,"\n")

        print(bcolors.UNDERLINE + bcolors.GREEN + "Mean Squared Error:" + bcolors.ENDC)
        print(mse,"\n")

        print(bcolors.UNDERLINE + bcolors.RED + "R2 score:" + bcolors.ENDC)
        print(r2,"\n")
        print(bcolors.YELLOW  + "="*min(len(var_b), int(columns)) + bcolors.ENDC)

        return ""

    def PolyDegree(self, Degree = 3):
        self.degree = Degree

class OLS(LinReg):

    def compute(self, XY):
        self.beta = np.linalg.inv(self.XY.T.dot(self.XY)).dot(self.XY.T).dot(self.zTrain)
        #self.zpredict = self.XY.dot(self.beta)

    def statistics(self):
        super().statistics()
        self.var_b = np.diag(np.linalg.inv(self.XY.T.dot(self.XY))*self.var_z)


class Ridge(LinReg):
    def compute(self, Degree):
        self.degree = Degree
        self._rearrange()
        self.I = np.identity(XY.shape[1])
        self.beta = np.linalg.inv(self.XY.T.dot(self.XY) + lamb*I).dot(self.XY.T).dot(self.zTrain)
        self.zpredict = XY.dot(self.beta)


    def statistics(self):
        super().statistics()
        XY2 = self.XY.T.dot(self.XY)
        W = np.linalg.inv(XY2 + lamb*self.I).dot(XY2)
        self.var_b = np.diag(W.dot(np.linalg.inv(XY2)).dot(W.T))*self.var_z
