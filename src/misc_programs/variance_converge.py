import numpy as np
import matplotlib.pylab as plt

N = 100
#deg = 2

x = np.random.uniform(0, 1, (N, 1))
y = np.random.uniform(0, 1, (N, 1))
noise = 0.2*np.random.randn(N, 1)
z = 2*x**2 + 3*y**2 + noise

XY = np.c_[np.ones(N), x, y, x**2, x*y, y**2]

beta = np.linalg.pinv(XY) @ z
zpredict = XY @ beta
varz = 1.0/(N - 3)*np.sum((z - zpredict)**2)
var_exact = np.diag(np.linalg.pinv(XY.T @ XY)*varz)

#print(beta)
#print(var_exact)

n = 6
var_diff = np.zeros(n)
Boots = np.array([10**i for i in range(n)])

c=0
for nBoots in Boots:

	betas = np.zeros((nBoots, XY.shape[1]))

	for i in range(nBoots):
		idx = np.random.choice(N, N)
		XYboot = XY[idx]
		zboot = z[idx]

		beta_boot = np.linalg.pinv(XYboot) @ zboot
		betas[i] = beta_boot.flatten()

	var_boot = np.var(betas, axis = 0)
	var_diff[c] = 1.0/n*np.sum(np.abs(var_exact - var_boot))

	c += 1

plt.semilogx(Boots, var_diff)
plt.xlabel("nBoots")
plt.ylabel(r"avg($|Var(\beta_{exact}) - Var(\beta_{est})|$)")
plt.show()



