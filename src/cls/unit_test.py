import numpy as np
from cls_reg import LinReg
from Franke import FrankeFunction
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso

def test_design_matrix():
    """ Checks that the design matrix made by the self-written code matches
    the one made by scikit learn, within a tolerance"""

    N = 100
    degrees = [2, 10, 20]

    x = np.random.uniform(0, 1, (N, 1))
    y = np.random.uniform(0, 1, (N, 1))
    z = x + y

    for deg in degrees:

        a = LinReg(x, y, z, deg)
        XY = a.XY

        poly = PolynomialFeatures(degree = deg)
        XY_sk = poly.fit_transform(np.append(x, y, axis=1))

        assert np.allclose(XY, XY_sk)


def test_ols():
    """ Checking that the implemented Ordinary Least Squared method gives the
    same results as the scikit version for a few different functions,
    within a tolerance."""


    N = 100
    deg = 3

    x = np.random.uniform(0, 1, (N, 1))
    y = np.random.uniform(0, 1, (N, 1))

    z1 = 2*x**2 + 2*x*y
    z2 = 3*y**4
    z3 = FrankeFunction(x, y)

    for z in [z1, z2, z3]:

        a = LinReg(x, y, z, deg)
        beta = a.ols()

        poly = PolynomialFeatures(degree = deg)
        XY_sk = poly.fit_transform(np.append(x, y, axis=1))
        Lin = LinearRegression(fit_intercept= False)
        Lin.fit(XY_sk, z)

        beta_sk = Lin.coef_.reshape(-1, 1)

        assert np.allclose(beta, beta_sk)


def test_ridge():
    """ Checking that the implemented Ordinary Least Squared method gives the
    same results as the scikit version for a few different functions,
    within a tolerance."""


    N = 100
    deg = 3
    lamb = 1.0

    x = np.random.uniform(0, 1, (N, 1))
    y = np.random.uniform(0, 1, (N, 1))

    z1 = 2*x**2 + 2*x*y
    z2 = 3*y**4
    z3 = FrankeFunction(x, y)

    for z in [z1, z2, z3]:

        a = LinReg(x, y, z, deg)
        a.lamb = lamb
        beta = a.ridge()

        poly = PolynomialFeatures(degree = deg)
        XY_sk = poly.fit_transform(np.append(x, y, axis=1))
        ridge = RidgeCV([lamb], fit_intercept= False)
        ridge.fit(XY_sk, z)

        beta_sk = ridge.coef_.reshape(-1, 1)

        assert np.allclose(beta, beta_sk)
