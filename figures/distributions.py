import chaospy as cp
import numpy as np
import pylab as plt

dist = cp.MvNormal([175,175],[[10,6],[6,10]])
t, s = np.linspace(165,185,100), np.linspace(165,185,100)
i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]

plt.figure()
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("mvnormal.png")

dist = cp.MvLognormal([2,2],[[1,0],[0,1]])
t, s = np.linspace(0,15,100), np.linspace(0,15,100)
i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]


plt.figure()
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("lognormal.png")


dist = cp.Iid(cp.Gamma(1,1),2)
#dist = cp.J(dist[0],dist[1])
t, s = np.linspace(0,2,100), np.linspace(0,2,100)
i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]


plt.figure()
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("gamma.png")



dist = cp.Iid(cp.Normal(1,1),2)
#dist = cp.J(dist[0],dist[1])
t, s = np.linspace(0,2,100), np.linspace(0,2,100)
i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]


plt.figure()
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("normal.png")


dist = cp.Iid(cp.Uniform(0.5,1.5),2)
#dist = cp.J(dist[0],dist[1])
t, s = np.linspace(0,2,100), np.linspace(0,2,100)
i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]


plt.figure()
plt.contourf(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("uniform.png")


dist = cp.Normal()
#dist = cp.J(dist[0],dist[1])
t = np.linspace(-3,3,100)


plt.figure()
plt.plot(t,dist.pdf(t),linewidth=4)
plt.savefig("uniform2.png")


dist = cp.Normal(1,0.5)
#dist = cp.J(dist[0],dist[1])
t = np.linspace(-3,3,100)


plt.figure()
plt.plot(t,dist.pdf(t),linewidth=4)
plt.savefig("uniform3.png")




def dirichlet(D):
    alpha = np.array([1./(D+1)]*(D+1))
    beta = []
    beta.append(cp.Beta(alpha[0],sum(alpha[1:])))
    c = beta[0]
    for i in range(1,D+1):
        beta.append(cp.Beta(alpha[i],sum(alpha[i+1:]))*(1-c))
        c = beta[i] + c
 
    result = beta[0]
    for b in beta[1:]:
        result = cp.J(result, b)
    return result
    

