"""
Testing the singular value decomosition
"""


import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.linalg as scl

N = 5
x = np.random.uniform(0, 1, (N,1))
y = np.random.uniform(0, 1, (N,1))
z = 2*x + 3*y**2

linreg = LinearRegression(fit_intercept=False)
X = np.c_[np.ones(N), x, y, x**2, x*y, y**2]

print("det(X^T @ X) = \n", scl.det(X.T @ X))

beta_inv = scl.inv(X.T @ X) @ X.T @ z
print("\nbeta inv : \n", beta_inv)
zpred_inv = X @ beta_inv


beta_svd = np.linalg.pinv(X) @ z
print("\nbeta svd : \n", beta_svd)
zpred_svd = X @ beta_svd


linreg.fit(X, z)
beta_sk = linreg.coef_
print("\nbeta scikit : \n", beta_sk.T)
zpred_sk = linreg.predict(X)

mse_inv = 1/N*np.sum((zpred_inv - z)**2)
mse_svd = 1/N*np.sum((zpred_svd - z)**2)
mse_sk = 1/N*np.sum((zpred_sk - z)**2)
print("\nMSE inv: ", mse_inv)
print("MSE SVD: ", mse_svd)
print("MSE SCIKIT: ", mse_sk)

print(scl.inv(X.T @ X))
