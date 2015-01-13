import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2
N = 6

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


#def u(t,a, I):
#    return I*np.exp(-a*t)

def u(x,a, I):
    #ax = np.outer(a,x)
    return I*np.exp(-a*x)
 
a = cp.Beta(2,2,0, 0.1)
I = cp.Beta(2,2,8, 10)
dist = cp.J(a,I)
T = np.linspace(0, 10, Nt+1)[1:]
dt = T[1] - T[0]


errorG = []
varG = []

P = cp.orth_ttr(7, dist)
nodes, weights = cp.generate_quadrature(8, dist, rule="G")
i1,i2 = np.mgrid[:len(weights), :Nt]
solves = u(T[i2],nodes[0][i1],nodes[1][i1])
U_analytical = cp.fit_quadrature(P, nodes, weights, solves)
    


Kl = []
for n in xrange(1,N+1):
     
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="G")
    Kl.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorG.append(dt*np.sum(np.abs(cp.E(U_analytical,dist) - cp.E(U_hat,dist))))
    varG.append(dt*np.sum(np.abs(cp.Var(U_analytical,dist) - cp.Var(U_hat,dist))))


errorL = []
varL = []
Ke = []
for n in xrange(1,N+1):
     
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="E")
    Ke.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorL.append(dt*np.sum(np.abs(cp.E(U_analytical,dist) - cp.E(U_hat,dist))))
    varL.append(dt*np.sum(np.abs(cp.Var(U_analytical,dist) - cp.Var(U_hat,dist))))

print cp.Var(U_analytical,dist)
print cp.Var(U_hat,dist)

pl.plot(Ke,errorG,linewidth=2)
pl.plot(Ke, varG,linewidth=2)
pl.plot(Kl,errorL,linewidth=2)
pl.plot(Kl, varL,linewidth=2)
pl.xlabel("Samples, k")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["E, Gaussian","Var, Gaussian","E, Legendre","Var, Legendre"])
pl.savefig("convergence_GvsL.png")

pl.show()
