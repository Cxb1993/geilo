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


def u(x,a, I):
    return I*np.exp(-a*x)


C = [[2,1],[1,2]]
mu = np.array([0.5,0.5])

#C = np.array([[1,0],[0.,1]])
#mu = np.array([0,0])

R = cp.J(cp.Normal(),cp.Normal())
#R = cp.J(cp.Uniform(),cp.Uniform())
L =  np.linalg.cholesky(C)

Q1 = cp.MvNormal(mu,C)
Q = R*L + mu 

def T(r):
    result = np.dot(L,r)
    i=0
    for m in mu:
        result[i] += mu[i]
        i+=1
    return result

"""
sample_R = R.sample(1000)
plt.scatter(*sample_R)
plt.show()
plt.scatter(*T(sample_R))
plt.show()
"""
x = np.linspace(0, 1, 100)
dt = x[1]-x[0]
M = 10

P = cp.orth_ttr(M, R)
nodes, weights = cp.generate_quadrature(M+1, R, rule="G")
nodes_ = T(nodes)
solves = [u(x, s[0], s[1]) for s in nodes_.T]
U_analytic, c = cp.fit_quadrature(P, nodes_, weights, solves, retall=True)



P = cp.orth_ttr(M, R)
nodes = R.sample(2*len(P), "S")
nodes_ = T(nodes)
#nodes_ = nodes
solves = [u(x, s[0], s[1]) for s in nodes_.T]
U_analytic = cp.fit_regression(P, nodes_, solves,rule="LS")


N = 8
error = []
var = []
K = []
for n in range(1,N+1):

    P = cp.orth_ttr(n, R)
    nodes = R.sample(2*len(P), "S")
    K.append(2*len(P))
    nodes_ = T(nodes)
    solves = [u(x, s[0], s[1]) for s in nodes_.T]
    U_hat = cp.fit_regression(P, nodes_, solves,rule="T")
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,R) - cp.E(U_hat,R))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,R) - cp.Var(U_hat,R))))
plt.rc("figure", figsize=[6,4])

plt.plot(K,error,"r-",linewidth=2)
plt.plot(K, var,"r--",linewidth=2)
plt.xlabel("Nodes, K")
plt.ylabel("Error")
plt.yscale('log')
plt.xlim([6,90])
#plt.title("Error in expectation value and variance ")
plt.legend(["Mean","Variance"])
plt.savefig("convergence_dependence.png")
plt.show()
