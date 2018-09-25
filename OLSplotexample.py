import numpy as np
import matplotlib.pyplot as plt


x = np.random.randn(10)
y = 2*x
y2 = y + np.random.randn(10)
z = y - y2
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.plot(x, y, label='RealDistribution', color='red')
ax.plot(x, y2, 'o', label='TrainingData', color='green')
plt.errorbar(x, y, yerr=[z,0*z], label='Fitted line with error', color='yellow', ecolor='red')
ax.legend()
plt.savefig('fig/OLS_RSS_example.png')
plt.show()

