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

# print(a.statistics().__dict__)
# print(a.__dict__)
# print(a)

# x = np.random.randn(100)
# print(np.average(x))
# def bootstrap(nBoots = 1000000):
#         bootVec = np.zeros(nBoots)
#         for k in range(0,nBoots):
#             bootVec[k] = np.average(np.random.choice(x, len(x)))
#         bootAvg = np.average(bootVec)

#         bootVar = np.var(bootVec)
#         bootStd = np.std(bootVec)
#         return bootVec

# bootVec = bootstrap()
# bootAvg = np.average(bootVec)
# print(bootAvg)

