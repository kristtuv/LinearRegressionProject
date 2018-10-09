import numpy as np
from cls.cls_reg import LinReg
from cls.Franke import FrankeFunction
import matplotlib.pylab as plt
#import plotparams

N = 100
degrees = range(10)
#degrees = [5]
#lambdas = [10**i for i in range(-3, 3)]
lambdas = [0.001, 0.01, 0.1]
#lambdas = [0.1]

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.3*np.random.randn(N, 1)
z = FrankeFunction(x, y) + noise

conf = np.zeros((len(lambdas), len(degrees)))


print("LASSO REGRESSION")
print("| Degree | lambda | MSE Train | MSE Test |  R2 Train | R2 Test | Conf.int.|")
print("="*75)
for i in range(len(degrees)):

    a = LinReg(x, y, z, degrees[i])

    for j in range(len(lambdas)):

        a.lamb = lambdas[j]
        beta = a.lasso(a.XY_Train, a.z_Train)
        zpred_train = a.XY_Train @ beta
        zpred_test = a.XY_Test @ beta

        mse_train = a.MSE(a.z_Train, zpred_train)
        mse_test = a.MSE(a.z_Test, zpred_test)
        r2_train  = a.R2(a.z_Train, zpred_train)
        r2_test  = a.R2(a.z_Test, zpred_test)

        a.bootstrap(100, a.lasso)
        conf_int = np.average(1.96*np.sqrt(a.betas_var))
        a.beta

        conf[j, i] = conf_int

        if degrees[i] == 3:
            a.lamb = 0.1
            beta_ridge = a.ridge(a.XY_Train, a.z_Train)
            beta_ols = a.ols(a.XY_Train, a.z_Train)
            print("\nOLS:\n", beta_ols.flatten())
            print("\nRIDGE:\n", beta_ridge.flatten())
            print("\nLASSO:\n", beta.flatten())

        print("|%8i|%8g|%11f|%10f|%10f|%10f|%10f|" % \
            (degrees[i], lambdas[j], mse_train, mse_test, r2_train, r2_test, conf_int))
    print("-"*75)

labels = [r"$\lambda=$" + str(x) for x in lambdas]

plt.plot(degrees, conf.T)
plt.xlabel("Complexity")
plt.ylabel("CI(95%)")
plt.legend(labels)
plt.show()

lambdas = np.array([0.01*i + 0.001 for i in range(20)])
noises = np.array([0.05*i for i in range(20)])
mses = np.zeros((len(noises), len(lambdas)))
deg = 3

for i in range(len(noises)):
    z = FrankeFunction(x, y) + noises[i]
    a = LinReg(x, y, z,deg)

    for j in range(len(lambdas)):

        a.lamb = lambdas[j]
        beta = a.lasso(a.XY_Train, a.z_Train)
        zpred_test = a.XY_Test @ beta
        mse_test = a.MSE(a.z_Test, zpred_test)
        mses[i, j] = mse_test

Noises, Lambdas = np.meshgrid(noises, lambdas)
extent = np.min(lambdas), np.max(lambdas), np.min(noises), np.max(noises)

plt.imshow(mses, aspect = "auto", extent = extent)
plt.colorbar()
plt.title(r"MSE($\lambda$, $\sigma$), Lasso Deg %i" %(deg), fontsize=16)
plt.xlabel(r"$\lambda$", fontsize=14)
plt.ylabel(r"$\sigma$", fontsize=14)
plt.show()
