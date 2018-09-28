import numpy as np
from cls_reg import LinReg

N = 100
deg = 1

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.5*np.random.randn(N, 1)
z = 2*x**2 + 3*y**2 + noise

a = LinReg(x, y, z, deg)
"""
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
"""

XY_Train, XY_Test, z_Train, z_Test = a.split_data(frac = 0.3)

beta = a.ols(XY_Train, z_Train)
zpred_Train = XY_Train @ beta
zpred_Test = XY_Test @ beta

print("MSE Train: ", a.MSE(z_Train,zpred_Train))
print("MSE Test:, ", a.MSE(z_Test, zpred_Test))
print("R2 Train: ", a.R2(z_Train,zpred_Train))
print("R2 Test: ", a.R2(z_Test, zpred_Test))
