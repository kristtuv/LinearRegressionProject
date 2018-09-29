import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
import plotparams

N = 1000
degrees = range(9)
#degrees = [2, 10]

x = np.random.uniform(-1, 1, (N, 1))
y = np.random.uniform(-1, 1, (N, 1))
noise = 0.4*np.random.randn(N, 1)
var_noise = np.var(noise)
print(var_noise)
#z = 2*x**2 + 3*y**2 + noise
#z = FrankeFunction(x, y) + noise
z_true = x**8 + y**6
# z_true = np.sin(x*np.pi)
z = z_true + noise

mse_Train = np.zeros(len(degrees))
mse_Test = np.zeros(len(degrees))
bias_Train = np.zeros(len(degrees))
bias_Test = np.zeros(len(degrees))
var = np.zeros(len(degrees))



for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    a.split_data(frac = 0.3)
    #a.lamb = 5.0

    XY_Train = a.XY_Train ; XY_Test = a.XY_Test
    z_Train = a.z_Train ; z_Test = a.z_Test

    z_avg, z_var, train_error, test_error = a.bootstrap(5000, a.ols)

    beta = a.ols(XY_Train, z_Train)
    zpred_Train = XY_Train @ beta
    zpred_Test = XY_Test @ beta

    #bias_Train[i] = np.average(z_Train - z_avg)**2
    bias_Test[i] = np.average((z_Test - z_avg)**2)
    var[i] = np.average(z_var)

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
#plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, bias_Test, label="Test Bias")
#plt.plot(degrees, bias_Test, label="Test Bias")
#plt.ylim([0, 1])
plt.legend()
plt.show()

#plt.plot(degrees, var_Train, label="Train Variance")
plt.plot(degrees, var, label="Variance")
#plt.ylim([0, 1])
plt.legend()
plt.show()

plt.plot(degrees, mse_Test, label = "Test error")
plt.plot(degrees, bias_Test, label = "Test Bias")
plt.plot(degrees, var, label = "Variance")
plt.plot(degrees, bias_Test + var - var_noise, label = "Bias + Var - noise")
#plt.ylim([0,1])
plt.legend()
plt.show()
