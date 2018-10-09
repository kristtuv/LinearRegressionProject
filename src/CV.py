"""
Running Cross-validation and bootstrap 
on the Franke function
"""


import numpy as np
from cls.cls_reg import LinReg
from cls.Franke import FrankeFunction
import matplotlib.pylab as plt
import cls.plotparams

def CV_models(x, y, z, degrees, nfolds, regressionmethod, lambdas=[0], plot= False):

    mse_Train = np.zeros(len(lambdas))
    mse_Test = np.zeros(len(lambdas))
    r2_Train = np.zeros(len(lambdas))
    r2_Test = np.zeros(len(lambdas))

    #method = str(regrssionmethod).split(".")[1]

    print("\n\n%s REGRESSION:\n" %(regressionmethod.upper()))
    print("="*60)
    print("\n| Degree | Lambda | MSE Train | MSE Test | R2 Train | R2 Test |")
    print("-"*60)

    for deg in degrees:

        a = LinReg(x, y, z, deg)
        method = getattr(a, regressionmethod)

        for i in range(len(lambdas)):

            a.lamb = lambdas[i]

            mse_train, mse_test, r2_train, r2_test = a.kfold(10, method)

            mse_Train[i] = mse_train
            mse_Test[i] = mse_test
            r2_Train[i] = r2_train
            r2_Test[i] = r2_test


            #err_diff = r2_test - mse_test
            err_diff = 1 - mse_test

            if np.any(best_models_err < err_diff):
                idx = np.argmax(best_models_err < err_diff)
                best_models_err[:] = np.insert(best_models_err, idx, err_diff)[:nbest]
                best_models.insert(idx, [regressionmethod.capitalize(), deg, lambdas[i], mse_test, r2_test])
                del best_models[nbest]


            print("|%8i|%8g|%11f|%10f|%10f|%10f|" % (deg, lambdas[i], mse_train, mse_test, r2_train, r2_test))

        if plot:
            plt.plot(np.log10(lambdas), mse_Train, label="MSE Train")
            plt.plot(np.log10(lambdas), mse_Test, label="MSE Test")
            plt.xlabel(r"$log_{10}\lambda$")
            plt.ylabel("Mean Squared Error")
            plt.title("%s regression, polynomial deg: %i" % (regressionmethod.capitalize(), deg))
            plt.legend()
            plt.show()

        print("-"*64)


N = 100
nfolds = 10
degrees = np.arange(10, dtype=np.int)
nbest = 15

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.3*np.random.randn(N, 1)
var_noise = np.var(noise)
#z = 2*x**2 + 3*y**2 + noise
z = FrankeFunction(x, y) + noise
#z = x**5 + y**5 + noise

lambdas_ridge = [10**i for i in range(-3, 3)]
lambdas_lasso = [10**i for i in range(-5, 1)]

best_models = [[]]*nbest
best_models_err = np.array([-10000]*nbest, dtype=np.float)

CV_models(x, y, z, degrees, nfolds, "ols")
CV_models(x, y, z, degrees, nfolds, "ridge", lambdas_ridge)
CV_models(x, y, z, degrees, nfolds, "lasso", lambdas_lasso)


print("\nBootstrapping best models:")
for i in range(nbest):

    model = LinReg(x, y, z, best_models[i][1])
    model.lamb = best_models[i][2]
    method = getattr(model, best_models[i][0].lower())
    bias, variance, train_error, test_error = model.bootstrap(100, method)

    best_models[i].extend([bias, variance, test_error])


print("\n\n Best Models:")
print("="*75)
print("     Method | Degree | lambda | MSE Test | R2 Test  |   Bias   | Variance | MSE Bootstrap|")
print("-"*75)
for i in range(nbest):
    print("%3i:%8s|%8i|%8g|%10g|%10f|%10f|%10f|%10f|" % (tuple([i+1] + best_models[i])))

#print("\nVariance of noise: ", var_noise)
