import numpy as np
from cls_reg import LinReg

N = 100
deg = 1

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.5*np.random.randn(N, 1)
z = 2*x**2 + 3*y**2 + noise

a = LinReg(x, y, z, deg)

beta = a.ols()
zpredict = a.XY @ beta

mse = a.MSE(z, zpredict)
print("MSE = ", mse)


model_mean = np.average(zpredict)
model_var = np.var(zpredict)
model_bias = 1.0/N*np.sum((z - model_mean)**2)
rest = 2.0/N*np.sum((z - model_mean)*(model_mean - zpredict))

print("Model variance: ", model_var)
print("Model bias: ", model_bias)
print("Variance + Bias: ", model_var + model_bias, "\n")
print("Rest: ", rest)

#a.kfold(5, a.ols)


"""
a.bootstrap(10000, a.ols)
print("\nBootstrap MSE: ", a.boot_mse)

zboot = a.XY @ a.beta_boot
MSE_boot = a.MSE(z, zboot)

boot_mean = np.average(zboot)
boot_var = np.var(zboot)
boot_bias = 1.0/N*np.sum((z - boot_mean)**2)

print("Boot variance: ", boot_var)
print("Boot bias: ", boot_bias)
print("Variance + Bias (Boot): ", boot_var + boot_bias, "\n")
#print("Rest: ", rest)

#print(model_var + model_bias)
#print(model_var + model_bias + rest)
"""
