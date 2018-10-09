import matplotlib.pyplot as plt
import numpy as np
import cls.plotparams
from cls.cls_reg import LinReg
from cls.split_patches import split_patches
from imageio import imread
import time
from tqdm import tqdm


############################
#Spliting data into patches
############################
z = imread('../data/n58_e006_1arc_v3.tif')[:-1, :-1]
z = z/np.max(z)
x = np.linspace(0,1, z.shape[0])
y = np.linspace(0,1, z.shape[1])
print(z.shape)
print(x.shape)
print(y.shape)

start = time.time()
num_patches = 5
x, y, z = split_patches(x, y, z, 36, 18, num_patches)
stop = time.time()
print('Split patches time: ', stop - start)




degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def lasso():
    "Run plotter for lasso mse vs degree"
    lambdas_lasso = [10**i for i in range(-5, 1)]
    for lamda in tqdm(lambdas_lasso):
        mse_Train = []
        mse_Test = []
        for deg in degrees:
            linreg = LinReg(x, y, z, deg)
            linreg.lamb = lamda
            mse_train, mse_test, r2_train, r2_test = linreg.kfold(num_patches, linreg.lasso)
            mse_Train.append(mse_train)
            mse_Test.append(mse_test)

        plt.figure()
        plt.title(r'Method:Lasso, $\lambda$: %s'%(lamda))
        plt.xlabel('Polynomial degrees')
        plt.ylabel('Mean Squared error')
        plt.plot(mse_Test, label = 'Test')
        plt.plot(mse_Train, label = 'Train')
        plt.legend()
        #plt.savefig('../plots/realdata/Lasso_lamb%s_scaled.png'%(lamda))
        plt.close()
def ridge():
    "Run plotter for ridge mse vs degree"
    lambdas_ridge = [10**i for i in range(-3, 3)]
    for lamda in tqdm(lambdas_ridge):
        mse_Train = []
        mse_Test = []
        for deg in degrees:
            linreg = LinReg(x, y, z, deg)
            linreg.lamb = lamda
            mse_train, mse_test, r2_train, r2_test = linreg.kfold(num_patches, linreg.ridge)
            mse_Train.append(mse_train)
            mse_Test.append(mse_test)

        plt.figure()
        plt.title(r'Method:Ridge, $\lambda$: %s'%(lamda))
        plt.xlabel('Polynomial degrees')
        plt.ylabel('Mean Squared error')
        plt.plot(mse_Test, label = 'Test')
        plt.plot(mse_Train, label = 'Train')
        plt.legend()
        #plt.savefig('../plots/realdata/Ridge_lamb%s_scaled.png'%(lamda))
        plt.close()
def ols():
    "Run plotter for ols mse vs degree"
    mse_Train = []
    mse_Test = []
    for deg in tqdm(degrees):
        linreg = LinReg(x, y, z, deg)
        mse_train, mse_test, r2_train, r2_test = linreg.kfold(num_patches, linreg.ols)
        mse_Train.append(mse_train)
        mse_Test.append(mse_test)

    plt.figure()
    plt.title('Method:Ols')
    plt.xlabel('Polynomial degrees')
    plt.ylabel('Mean Squared error')
    plt.plot(mse_Test, label = 'Test')
    plt.plot(mse_Train, label = 'Train')
    plt.legend()
    #plt.savefig('../plots/realdata/Ols_scaled.png')
    plt.close()

def lasso_lamb():
    "Run plotter for lasso mse vs lambda"
    degrees = [8, 9, 10, 11, 12]
    lambdas_lasso = [10**i for i in range(-5, 1)]
    for deg in degrees:
        mse_Train = []
        mse_Test = []
        for lamda in tqdm(lambdas_lasso):
            linreg = LinReg(x, y, z, deg)
            linreg.lamb = lamda
            mse_train, mse_test, r2_train, r2_test = linreg.kfold(num_patches, linreg.lasso)
            mse_Train.append(mse_train)
            mse_Test.append(mse_test)

        plt.figure()
        plt.title(r'Method:Lasso, degree: %s'%(deg))
        plt.xlabel(r'log $\lambda$')
        plt.ylabel('Mean Squared error')
        plt.plot(np.log10(lambdas_lasso), mse_Test, label = 'Test')
        plt.plot(np.log10(lambdas_lasso), mse_Train, label = 'Train')
        plt.legend()
        #plt.savefig('../plots/realdata/Lasso_deg%s_scaled.png'%(deg))
        plt.close()
def ridge_lamb():
    "Run plotter for ridge mse vs lambda"
    degrees = [8, 9, 10, 11, 12]
    lambdas_ridge = [10**i for i in range(-3, 3)]
    for deg in degrees:
        mse_Train = []
        mse_Test = []
        for lamda in tqdm(lambdas_ridge):
            linreg = LinReg(x, y, z, deg)
            linreg.lamb = lamda
            mse_train, mse_test, r2_train, r2_test = linreg.kfold(num_patches, linreg.ridge)
            mse_Train.append(mse_train)
            mse_Test.append(mse_test)

        plt.figure()
        plt.title(r'Method:Ridge, deg: %s'%(deg))
        plt.xlabel(r'log $\lambda$')
        plt.ylabel('Mean Squared error')
        plt.plot(np.log10(lambdas_ridge), mse_Test, label = 'Test')
        plt.plot(np.log10(lambdas_ridge), mse_Train, label = 'Train')
        plt.legend()
        #plt.savefig('../plots/realdata/Ridge_deg%s_scaled.png'%(deg))
        plt.close()

if __name__=='__main__':
    lasso()
    ridge()
    ols()
    lasso_lamb()
    ridge_lamb()
