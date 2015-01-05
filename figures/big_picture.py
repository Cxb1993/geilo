import chaospy as cp
dist = cp.Normal()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

mean = 0.5
variance = 0.1
sigma = math.sqrt(variance)
x = np.linspace(-0.5,1.5,100)
x1 = np.linspace(-4,4,100)

plt.figure(1)
plt.plot(x,mlab.normpdf(x,mean,sigma),linewidth=3.0)
plt.savefig("gausian1.png")
#plt.clear()
plt.figure(2)
variance = 1
sigma = math.sqrt(variance)
plt.plot(x1,mlab.normpdf(x1,mean,sigma),"r",linewidth=3.0)
plt.savefig("gausian2.png")

plt.figure()
plt.plot(x1,2 - np.exp(-0.15*x1), "g",linewidth=3.0)
plt.ylim([-0.5,1.5])
plt.savefig("gausian3.png")



plt.show()
