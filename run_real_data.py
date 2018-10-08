import matplotlib.pyplot as plt
import numpy as np
import plotparams
from cls_reg import LinReg
from split_patches import split_patches
from imageio import imread
import time



z = imread('data/n58_e006_1arc_v3.tif')[:-1, :-1]
x = np.linspace(0,1, z.shape[0])
y = np.linspace(0,1, z.shape[1])
print(z.shape)
print(x.shape)
print(y.shape)

start = time.time()
num_patches = 100   
x, y, z = split_patches(x, y, z, 360, 180, num_patches)
stop = time.time()
print('Split patches time: ', stop - start)




degrees = [0, 1, 2, 3, 4, 5, 6, 7, 8]

def lasso():
    lambdas_lasso = [10**i for i in range(-5, 1)]
    for lamda in lambdas_lasso:
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
        plt.savefig('plots/realdata/Lasso_lamb%s.png'%(lamda))
        plt.close()
def ridge():
    lambdas_ridge = [10**i for i in range(-3, 3)]
    for lamda in lambdas_ridge:
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
        plt.savefig('plots/realdata/Ridge_lamb%s.png'%(lamda))
        plt.close()
def ols():
    mse_Train = []
    mse_Test = []
    for deg in degrees:
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
    plt.savefig('plots/realdata/Ols.png')
    plt.close()

if __name__=='__main__':
    lasso()
    ridge()
    ols()

