import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
import matplotlib.pylab as plt
import plotparams

N = 100
nfolds = 10
degrees = range(5)

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.2*np.random.randn(N, 1)
var_noise = np.var(noise)
#z = 2*x**2 + 3*y**2 + noise
#z = FrankeFunction(x, y) + noise
#z_true = x**8 + y**6
z = x**4 + x**3*y + noise

lambdas = [10**i for i in range(-3, 3)]

mse_Train = np.zeros(len(lambdas))
mse_Test = np.zeros(len(lambdas))
r2_Train = np.zeros(len(lambdas))
r2_Test = np.zeros(len(lambdas))

print("\n\nRIDGE REGRESSION:\n")
print("="*60)
print("\n| Degree | Lambda | MSE Train | MSE Test | R2 Train | R2 Test |")
print("-"*60)

for deg in degrees:

    a = LinReg(x, y, z, deg)

    for i in range(len(lambdas)):

        a.lamb = lambdas[i]

        mse_train, mse_test, r2_train, r2_test = a.kfold(10, a.ridge)

        mse_Train[i] = mse_train
        mse_Test[i] = mse_test
        r2_Train[i] = r2_train
        r2_Test[i] = r2_test

        print("|%8i|%8g|%11f|%10f|%10f|%10f|" % (deg, lambdas[i], mse_train, mse_test, r2_train, r2_test))
    """
    plt.plot(np.log10(lambdas), mse_Train, label="MSE Train")
    plt.plot(np.log10(lambdas), mse_Test, label="MSE Test")
    plt.xlabel(r"$log_{10}\lambda$")
    plt.ylabel("Mean Squared Error")
    plt.title("Ridge regression, polynomial deg: %i" % deg)
    plt.legend()
    plt.show()
    """
    print("-"*64)
