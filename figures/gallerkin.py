import chaospy as cp
import pylab as pl
import numpy as np
from scipy.integrate import ode 
from math import factorial


def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)

a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)

x = np.linspace(0, 10, 101)[1:]
dt = x[1] - x[0]

M = 4
D = 2
N = factorial(M+D)/factorial(M) - 1

N = 4

#gaussian quadrature
error = []
var = []
K = []
for n in xrange(1,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="G")
    K.append(len(nodes[0]))
    solves = [u(x, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    error.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,dist))))
    
pl.figure()
pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)


N = 6

#Point collocation
error = []
var = []
K = []
for n in range(1,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "M")
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,dist))))


pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)







#Intrusive Gallerkin

n = 2
P, norm = cp.orth_ttr(n, dist, retall=True)
nodes, weights = cp.generate_quadrature(n+1, dist, rule="G")
solves = [u(x, s[0], s[1]) for s in nodes.T]
U_hat, c = cp.fit_quadrature(P, nodes, weights, solves, retall=True)

N = len(P)
q0, q1 = cp.variable(2)

P_nk = cp.outer(P, P)
E_ank = cp.E(q*P_nk,dist)
E_nk = cp.E(P_nk,dist)


print (E_ank/E_nk).shape
print E_nk.shape



pl.xlabel("Samples, k")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["E, GQ","Var, GQ", "E, PC","Var, PC,", "E, IG","Var, IG"])
#l.savefig("convergence_gallerkin.png")
pl.show()
