import chaospy as cp
import numpy as np
import pylab as plt
"""
 dist = cp.MvNormal([175,175],[[10,6],[6,10]])
t, s = np.linspace(165,185,100), np.linspace(165,185,100)

i1, i2 = np.mgrid[:100,:100]
t = t[i1]; s = s[i2]
plt.contour(s, t,dist.pdf([t,s]),alpha=0.6)
plt.savefig("dependent.pdf")
#plt.show()
"""





C = [[2,1],[1,2]]
mu = [1.5,1.5]
Q1 = cp.MvNormal(mu,C)

R = cp.J(cp.Normal(),cp.Normal())
L =  np.linalg.cholesky(C)
Q = R*L + mu 

print Q
print Q1

def T(r):
    return r*L + mu

x = np.linspace(0, 10, 100)
M = 4

P = cp.orth_ttr(M, Q)
nodes, weights = cp.generate_quadrature(M+1, Q, rule="G")
solves = [u(x, T(s[0]), T(s[1])) for s in nodes.T]
U_analytic, c = cp.fit_quadrature(P, nodes, weights, solves, retall=True)







N = 3
error = []
var = []
K = []
for n in range(1,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "M")
    K.append(2*len(P))
    solves = [u(T, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))
