"""
Plotting mse bias variance decomosition
for OLS ont he franke function
"""

import numpy as np
from cls.cls_reg import LinReg
from cls.Franke import FrankeFunction
import matplotlib.pylab as plt
import cls.plotparams

N = 1000
degrees = range(11)

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.5*np.random.randn(N, 1)
var_noise = np.var(noise)
#z = 2*x**2 + 3*y**2 + noise
z = FrankeFunction(x, y) + noise
#z_true = x**8 + y**6
#z = x**4 + x**3*y + noise

mse_Train = np.zeros(len(degrees))
mse_Test = np.zeros(len(degrees))
bias_Train = np.zeros(len(degrees))
bias_Test = np.zeros(len(degrees))
var = np.zeros(len(degrees))



for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    a.split_data(frac = 0.3)
    a.lamb = 0.005

    XY_Train = a.XY_Train ; XY_Test = a.XY_Test
    z_Train = a.z_Train ; z_Test = a.z_Test

    bias, variance, train_error, test_error = a.bootstrap(1000, a.ols)

    beta = a.ols(XY_Train, z_Train)
    zpred_Train = XY_Train @ beta
    zpred_Test = XY_Test @ beta

    #bias_Train[i] = np.average(z_Train - z_avg)**2
    bias_Test[i] = bias
    var[i] = variance

    #mse_Train[i] = a.MSE(z_Train,zpred_Train)
    #mse_Test[i] = a.MSE(z_Test, zpred_Test)

    mse_Train[i] = train_error
    mse_Test[i] = test_error

    print("\nErrors for pol deg %i:" %(degrees[i]))
    print("="*40)
    print("MSE Train: ", mse_Train[i])
    print("MSE Test: ", mse_Test[i])
    print("R2 Train: ", a.R2(z_Train,zpred_Train))
    print("R2 Test: ", a.R2(z_Test, zpred_Test))
    print("Bias Train: ", bias_Train[i])
    print("Bias Test: ", bias_Test[i])
    print("Variance: ", var[i])
    #print("Bias + var - mse = ", bias_Test[i] + var_Test[i] - mse_Test[i])
    #print("var noise = ", var_noise)

plt.plot(degrees, mse_Train, label="Train Error")
plt.plot(degrees, mse_Test, label="Test Error")
plt.xlabel("Complexity")
plt.ylabel("MSE")
#plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, bias_Test, label="Test Bias")
plt.xlabel("Complexity")
plt.ylabel("Bias")
#plt.plot(degrees, bias_Test, label="Test Bias")
#plt.ylim([0, 1])
plt.legend()
plt.show()

#plt.plot(degrees, var_Train, label="Train Variance")
plt.plot(degrees, var, label="Variance")
plt.xlabel("Complextiy")
plt.ylabel("Var")
#plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, mse_Test, label = "Test error")
plt.plot(degrees, bias_Test, label = "Bias")
plt.plot(degrees, var, label = "Variance")
plt.xlabel("Complexity")
plt.ylabel("Error")
#plt.ylim([0,1])
plt.legend()
plt.show()
