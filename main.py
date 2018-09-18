# from mlearning.LinRegSK import OLSsk, Ridgesk, Lassosk
from mlearning.franke import Franke
import numpy as np
from mlearning.LinReg import LinReg
from mlearning.LinReg import OLS
x = np.random.rand(10,1)
y = np.random.rand(10,1)
f = Franke(x,y).compute()
a = OLS(x,y,f)
a._testshit( 0.8)
# def bootstrap(self, nBoots = 1000):
#         bootVec = np.zeros(nBoots)
#         for k in range(0,nBoots):
#             bootVec[k] = np.average(np.random.choice(self.data, len(self.data)))
#         self.bootAvg = np.average(bootVec)
#         self.bootVar = np.var(bootVec)
#         self.bootStd = np.std(bootVec)
#         return(bootVec)
# plot(
# ab = np.append(x,y)

# # self.data[k] = np.average(np.random.choice(localDatacopy, len(localDatacopy)))
# a = OLS(x, y, f)
# a.compute(3)
# print(a.zpredict)
# a.statistics()
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


