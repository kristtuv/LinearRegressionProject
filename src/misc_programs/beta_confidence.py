"""
Generating beta plots with 
confidence intervals
"""
import sys
sys.path.append('../../')
sys.path.append('../')
import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
from matplotlib.ticker import MaxNLocator
import plotparams

N = 1500

x = np.random.uniform(0, 1, (N,1))
y = np.random.uniform(0, 1, (N,1))

for deg in [2, 5]:
    for sigma, noisy in zip([0, 0.5], ['', '_noisy']):
        noise = sigma*np.random.randn(N,1)
        #z = 2*x**3 + y**2 + noise
        z = FrankeFunction(x, y) + noise

        a = LinReg(x, y, z, deg)
        a.split_data(frac=0.1)
        beta_ols = a.ols(a.XY_Train, a.z_Train)
        zpredict = a.XY_Test @ beta_ols
        mse = a.MSE(a.z_Test, zpredict)
        var = np.diag(a.var_ols)
        conf = a.conf_ols
        #print(beta_ols)
        #print(var)
        #print(conf)
        print("MSE: ", mse)

        ax = plt.figure().gca()

        ax.errorbar(range(len(var)), beta_ols, yerr=conf, fmt='o', elinewidth=2.0, capsize=5.0)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(r'Deg: %i, $\sigma$ = %g' %(deg, sigma))
        ax.set_xlabel('j')
        ax.set_ylabel(r'$\beta_j$')
        # plt.savefig('plots/conf_ols'+str(deg)+noisy+'.png')
        plt.show()

"""
a.lamb = 1.0
beta_ridge = a.ridge()
print(beta_ridge)

a.lamb = 0.001
beta_lass = a.lasso()
print(beta_lass)
"""
