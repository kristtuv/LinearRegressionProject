import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
import plotparams

N = 1000
degrees = range(12)

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0*np.random.randn(N, 1)
var_noise = np.var(noise)
#z = 2*x**2 + 3*y**2 + noise
#z = FrankeFunction(x, y) + noise
z = 3*x**6 + 4*y**3 + 2*x*y**4 + noise

mse_Train = np.zeros(len(degrees))
mse_Test = np.zeros(len(degrees))
bias_Train = np.zeros(len(degrees))
bias_Test = np.zeros(len(degrees))
var_Train = np.zeros(len(degrees))
var_Test = np.zeros(len(degrees))


for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    XY_Train, XY_Test, z_Train, z_Test = a.split_data(frac = 0.3)

    beta = a.ols(XY_Train, z_Train)
    zpred_Train = XY_Train @ beta
    zpred_Test = XY_Test @ beta

    bias_Train[i] = np.average(zpred_Train - z_Train)**2
    bias_Test[i]  = np.average(zpred_Test - z_Test)**2
    var_Train[i] = np.var(zpred_Train - z_Train)
    var_Test[i] = np.var(zpred_Test - z_Test)

    mse_Train[i] = a.MSE(z_Train,zpred_Train)
    mse_Test[i] = a.MSE(z_Test, zpred_Test)

    print("\nErrors for pol deg %i:" %(degrees[i]))
    print("="*40)
    print("MSE Train: ", mse_Train[i])
    print("MSE Test: ", mse_Test[i])
    print("R2 Train: ", a.R2(z_Train,zpred_Train))
    print("R2 Test: ", a.R2(z_Test, zpred_Test))
    print("Bias + var - mse = ", bias_Test[i] + var_Test[i] - mse_Test[i])
    print("var noise = ", var_noise)

plt.plot(degrees, mse_Train, label="Train Error")
plt.plot(degrees, mse_Test, label="Test Error")
plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, bias_Train, label="Train Bias")
plt.plot(degrees, bias_Test, label="Test Bias")
#plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, var_Train, label="Train Variance")
plt.plot(degrees, var_Test, label="Test Variance")
#plt.ylim([0, 1])
plt.legend()
plt.show()
