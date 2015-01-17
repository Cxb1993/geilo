import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2
N = 50

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)
 
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)
T = np.linspace(0, 10, Nt+1)[1:]
dt = 10./Nt
pl.rc("figure", figsize=[6,4])

pl.figure(1)
pl.plot(-1,1, "k-")
pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "b")
#pl.plot(-1,1, "g")
pl.legend(["Mean","Variance", "Not nested","Nested","$L=M+1$"],loc=3,prop={"size" :12})
#pl.ylim([10**-14,10**2])



N = 5

errorCP = []
varCP = []
K = []
for n in xrange(0,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="E",sparse=True,growth=False)
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    


pl.plot(K,errorCP,"r-",linewidth=2)
pl.plot(K, varCP,"r--",linewidth=2)

"""

N = 5

errorCP = []
varCP = []
K = []
for n in xrange(1,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n, dist, rule="E")
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))

pl.plot(K,errorCP,"b-",linewidth=2)
pl.plot(K, varCP,"b--",linewidth=2)


N = 4
errorCP = []
varCP = []
K = []
for n in xrange(1,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="E")
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    
pl.plot(K,errorCP,"g-",linewidth=2)
pl.plot(K, varCP,"g--",linewidth=2)
#pl.ylim(10**-18, 10**2)
"""


pl.xlabel("Samples, K")
pl.ylabel("Error")
pl.yscale('log')
#pl.xlim([5,35])
#pl.ylim([10**-16,2*10**3])

#pl.title("Error in expectation value and variance ")
#pl.legend(["E, $L=M-1$","Var, $L=M-1$","Exp, $L=M$","Var,  $L=M$","E, $L=M+1$","Var,  $L=M+1$"],loc=3)
pl.savefig("convergence_2D_L.png")
















#pl.figure(2)
#pl.plot(-1,1, "r-")
#pl.plot(-1,1, "r--")
#pl.plot(-1,1, "r")
#pl.plot(-1,1, "b")
#pl.plot(-1,1, "g")
#pl.legend(["Mean","Variance", "$L=M-2$","$L=M-1$","$L=M$"],loc=3,prop={"size" :12})

#pl.rc("figure", figsize=[6,4])


N = 4
errorCP = []
varCP = []
K = []
for n in xrange(0,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="E",sparse=True,growth=True)
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    
pl.plot(K,errorCP,"b-", linewidth=2)
pl.plot(K, varCP,"b--",linewidth=2)

"""
N = 5
errorCP = []
varCP = []
K = []
for n in xrange(2,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n-1, dist, rule="E",sparse=True)
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    
pl.plot(K,errorCP,"b-",linewidth=2)
pl.plot(K, varCP,"b--",linewidth=2)


N = 4
errorCP = []
varCP = []
K = []
for n in xrange(2,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n, dist, rule="E",sparse=True)
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    
pl.plot(K,errorCP,"g-",linewidth=2)
pl.plot(K, varCP,"g--",linewidth=2)
"""


pl.xlabel("Samples, K")
pl.ylabel("Error")
pl.yscale('log')
pl.xlim([0,150])
pl.ylim([10**-16,2*10**2])

#Zpl.title("Error in expectation value and variance, sparse ")
#pl.legend(["E, $L=M-1$","Var, $L=M-1$","E, $L=M$","Var, $L=M$"],loc=2)
pl.savefig("convergence_2D_L_sparse.png")

pl.show()
