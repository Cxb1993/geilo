import chaospy as cp
import numpy as np
import pylab as plt


dist = cp.MvNormal([0.1,9],[[1,0.5],[0.5,1]])
sample = dist.sample(50)
t, s = np.linspace(-6,6,100), np.linspace(3,17,100)

i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.scatter(sample[1],sample[0])
plt.savefig("sampling.pdf")
plt.show()
