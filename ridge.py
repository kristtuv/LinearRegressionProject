import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
import plotparams

N = 100
degrees = range(6)

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
z = FrankeFunction(x, y)

print("RIDGE REGRESSION")
print("Degree| MSE | R2-Score | Variance")
for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    beta = a.ridge()
    zpredict = a.XY @ beta

    mse = a.MSE(z, zpredict)
    r2  = a.R2(z, zpredict)

    print(degrees[i], mse, r2, np.average(np.diag(a.var_ridge)))
