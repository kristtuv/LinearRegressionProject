import numpy as np
class LinReg:
    def __init__(self, x, y, Function):
        """ Trainingdata X and Y
            Function for fitting
            Fraction of TrainData set aside for testing
        """
        
        self.x = x
        self.y = y
        self.z = Function
        self.N = len(self.x)

               
    def _testshit(self,Test):
        self.Test=Test
        nTestData = int (np.floor(self.N*self.Test)) #Number of training data
        TestDataIndex = np.random.choice(range(self.N), nTestData,\
                replace = False)
        #Storing TestData
        self.TestDataY = self.x[TestDataIndex]
        self.TestDataX = self.y[TestDataIndex]
        self.TestDataZ = self.z[TestDataIndex]

        #Removing TestData from TraingData
        self.x = np.delete(self.x, TestDataIndex)
        self.y =  np.delete(self.y, TestDataIndex)
        self.z =  np.delete(self.z, TestDataIndex)
    def bootstrap(self, nBoots = 1000):
        print('not implemented yet')
        exit()
        localDatacopy = self.data.copy()
        # self.data = np.zeros(nBoots)
        self.x = np.zeros(nBoots)
        self.y = np.zeros(nBoots)
        for k in range(0,nBoots):
            self.data[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
            self.x[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
            self.y[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
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
        N = len(self.x)
        self.XY = np.ones([N, 1])
        for deg in range(1,self.degree + 1):
            liste = np.arange(deg+1)
            for i, j in zip(liste, np.flip(liste, 0)):
                col = self.x**(j)*self.y**i
                self.XY = np.append(self.XY, col, axis=1)

    def statistics(self):
        N = self.XY.shape[0]
        squared_error = np.sum((self.z - self.zpredict)**2)
        zmean = 1.0/N*np.sum(self.z)
        self.var_z = 1.0/(N - self.degree -1)*squared_error
        
        #Mean squared error
        self.mse = 1.0/N*squared_error
        #R2-score
        self.r2 = 1 - np.sum((self.z - self.zpredict)**2)/np.sum((self.z - zmean)**2)

class OLS(LinReg):

    def compute(self, Degree):
        self.degree = Degree
        self._rearrange()
        self.beta = np.linalg.inv(self.XY.T.dot(self.XY)).dot(self.XY.T).dot(self.z)
        self.zpredict = self.XY.dot(self.beta)

    def statistics(self):
        super().statistics()
        self.var_b = np.diag(np.linalg.inv(self.XY.T.dot(self.XY))*self.var_z)


        def __str__(self):
            pass

class Ridge(LinReg):
    def compute(self, Degree):
        self.degree = Degree
        self._rearrange()
        self.I = np.identity(XY.shape[1])
        self.beta = np.linalg.inv(self.XY.T.dot(self.XY) + lamb*I).dot(self.XY.T).dot(self.z)
        self.zpredict = XY.dot(self.beta)


    def statistics(self):
        super().statistics()
        XY2 = self.XY.T.dot(self.XY)
        W = np.linalg.inv(XY2 + lamb*self.I).dot(XY2)
        self.var_b = np.diag(W.dot(np.linalg.inv(XY2)).dot(W.T))*self.var_z

 
