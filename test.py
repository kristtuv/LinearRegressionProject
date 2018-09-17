import numpy as np
# a = np.arange(4)
# b = np.arange(4)
# for i in np.flip(a, 0):
#     for j in b:
#         print (j, i)

# for i, j in zip(a, b):
#     print (i+j, j)
# for i in range(len(a)):
#     for j in a:

# #         for k in b:
# counter = 0
# rang1= []
# while counter <= 3:
#     range1 = np.array(rang1)
#     for i, j in zip(rang1, np.flip(rang1, 0)):
#         print(j, i)
#     counter +=1
#     rang1.append(counter)

degree = 3 #Polynomial degree
for deg in range(degree + 1):
    liste = np.arange(deg+1)
    for i, j in zip(liste, np.flip(liste, 0)):
        print(j, i)
# deg = 2
# N = 10
# XY = np.ones([N, 1])
# N = 10
# x = np.random.rand(N,1)
# y = np.random.rand(N,1)
# count= 1
# for i in range(1,deg+1):
#     for j in range(deg+1):
#         if (i == 0) and (j==0):
#             continue
#         print("x^", i-j, "y^", j)
#         col = (x**i)*(y**j)
#         XY = np.append(XY, col, axis=1)
#         count += 1
# print("count: ", count)
