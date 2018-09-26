import numpy as np
from cls_reg import LinReg

N = 100
deg = 2

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.2*np.random.randn(N, 1)
z = 2*x**2 + 3*y**2 + noise
# print(type(x))
a = LinReg(x, y, z, deg)
a.bootstrap(100, a.ols)
# b = None
# print(type(b))

##a.lamb = 0.0001
##beta = a.lasso()
##print(beta)

##a.set_ols

# #beta = a.ols()
# # zpredict = a.XY @ beta


# # mse = a.MSE(z, zpredict)
# # r2 = a.R2(z, zpredict)
# # print("Initial MSE: ", mse)
# # print("Initial R2: ", r2)
# # print("\n")a.lamb = 0.0001

# # print("Initial betas: \n", beta)
# # print("Initial Var(beta): \n", np.diag(a.var_ols))
# # print("="*20)

##a.lamb = 0.0001
## a.kfold(5, a.ols)
## a.kfold(5, a.ridge)
## a.kfold(5, a.lasso)
##a.bootstrap(10000, a.ridge)


#"""
#a.ols()
#zpredict = a.XY @ a.beta_ols

#mse = 1.0/x.shape[0]*np.sum((z - zpredict)**2)
#print("Initial MSE: ", mse)

#a.bootstrap(1000)
#a.bootstrap(10000)
#"""

# import inspect
# def check_types(*args):
#     def decorator(func):
#         def wrapper(*argswrapper):
#             for a, b in zip(args, argswrapper):
#                 if a is not type(b):
#                     raise TypeError('See documentation for argument types')
#             return func(*argswrapper)
#         return wrapper
#     return decorator




# @check_types(int, float, int)
# def testfunc(a, b, c):
#     return 'MSE' 
# print(testfunc(1, 2, 3))





# def tags(tag_name):
#     def tags_decorator(func):
#         def func_wrapper(name):
#             return "<{0}>{1}</{0}>".format(tag_name, func(name))
#         return func_wrapper
#     return tags_decorator

# @tags("p")
# def get_text(name):
#     return "Hello "+name

# print (get_text("John"))
