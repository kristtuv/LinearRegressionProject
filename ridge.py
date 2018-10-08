import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
import plotparams

N = 1000
degrees = range(6)
#degrees = [5]
lambdas = [10**i for i in range(-3, 2)]
#lambdas = [0.1]

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.3*np.random.randn(N, 1)
z = FrankeFunction(x, y) + noise

print("RIDGE REGRESSION")
print("| Degree | lambda | MSE Train | MSE Test |  R2 Train | R2 Test | Variance |")
print("="*75)
for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    for lamb in lambdas:

        a.lamb = lamb
        beta = a.ridge(a.XY_Train, a.z_Train)
        zpred_train = a.XY_Train @ beta
        zpred_test = a.XY_Test @ beta

        mse_train = a.MSE(a.z_Train, zpred_train)
        mse_test = a.MSE(a.z_Test, zpred_test)
        r2_train  = a.R2(a.z_Train, zpred_train)
        r2_test  = a.R2(a.z_Test, zpred_test)

        print("|%8i|%8g|%11f|%10f|%10f|%10f|%10f|" % \
            (degrees[i], lamb, mse_train, mse_test, r2_train, r2_test, np.average(np.diag(a.var_ridge))))
    print("-"*75)
