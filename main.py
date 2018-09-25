# from mlearning.LinRegSK import OLSsk, Ridgesk, Lassosk
from mlearning.franke import Franke
import numpy as np
from mlearning.LinReg import LinReg
from mlearning.LinReg import OLS
x = np.random.rand(1000,1)
y = np.random.rand(1000,1)
f = Franke(x,y).compute()
a = OLS(x,y,f, 0.0)
a.PolyDegree(5)
print(a.__class__)
a.statistics()
print(a)
print(a.squared_error)
print(a.var_z)
# print(a.var_b)
# print(a.squared_error)
# print(a.var_z)
# print(a)

# print(a.var_b)

# b = Ridge(x, y, f)
# b.compute(3)
# b.statistics()
# # a = LinReg(x, y , f )
# # a.OLS(3)
# # print(a.beta)
# # print(x)
# # print()
# # print(a.data)
# # a.bootstrap()
# # print(
# # print(a.data)
# # # print(a.bootVec)
