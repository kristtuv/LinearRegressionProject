"""
Nasty program for generating a plot
showing the normal distribution-ish
of the bootstrap as nBoots increases
"""
import sys
sys.path.append('../../')
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import plotparams
N = 3
x = np.linspace(-5, 5, 15).reshape(-1, 1)
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(x)
linreg = LinearRegression()
yvecdata = np.zeros((N, x.shape[0]))

fig, ax = plt.subplots(2, 1, sharey=True)
ax[0].set_ylabel('Y')
ax[1].set_ylabel('Y')
print(np.random.randint(6))
for i in range(N):
    noise = np.random.randn(len(x)).reshape(-1,1)
    y = x**2 + 10*noise
    linreg.fit(X, y)
    ypredict= linreg.predict(X)
    yvecdata[i] = ypredict.flatten()
    ax[0].scatter(x, y)
    ax[0].plot(x, ypredict, linewidth=1)
ax[0].plot(x,np.mean(yvecdata, axis = 0).reshape(-1, 1), color = 'green', linewidth=2, label='Average')
ax[0].set_title('Random datasets')

N = 5
# x = np.linspace(-5, 5, 100gcc).reshape(-1, 1)
noise = np.random.randn(len(x)).reshape(-1,1)
y = x**2 + 10*noise
print(y.shape)
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(x)
print(X.shape)
linreg = LinearRegression()
yvecboot = np.zeros((N, x.shape[0]))
for i in range(N):
    choice = np.random.choice(x.shape[0], x.shape[0])
    linreg.fit(X[choice], y[choice])
    ypredict= linreg.predict(X)
    yvecboot[i] = ypredict.flatten()
    # if symbols[i] == '.':
    #     plt.scatter(x[choice], y[choice], marker=symbols[i],  color=colors[i], linewidth=0.8, s = 70)
    # else:
    #     plt.scatter(x[choice], y[choice], marker=symbols[i],  color=colors[i], linewidth=0.8, s = 500)
    ax[1].plot(x, ypredict, linewidth = 1)
ax[1].scatter(x, y, label='Original Data')
ax[1].plot(x,np.mean(yvecboot, axis = 0).reshape(-1, 1), color='green',linewidth=2, label='Average')
ax[1].set_title('Bootstrap')
plt.xlabel('X')
plt.tight_layout()
plt.savefig('plots/bootstrapexplain.png')
plt.show()
