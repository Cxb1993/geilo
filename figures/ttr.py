import chaospy as cp
import pylab as plt
import numpy as np
from math import factorial

def E_analytical(x):
    return 10*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 5*(1-np.exp(-0.2*x))/(x) - (10*(1-np.exp(-0.1*x))/(x))**2

    
a = cp.Uniform(0,0.1)
def u(x,a):
    ax = np.outer(a,x)
    return np.exp(-ax)



M = range(1,5)

a_m = cp.E(a)
Nx = 100
x = np.linspace(0.000000001, 10, Nx)
dt = x[1] - x[0]

legend = []

for m in M:
    K = range(1,10)

    Var = []
    E = []
    for k in K:
        P, norm = cp.orth_ttr(m, a, retall=True)
        N = len(P)
        nodes, weights = cp.generate_quadrature(k, a, rule="G")
        solves = u(x,nodes[0])
        U_hat, c = cp.fit_quadrature(P, nodes, weights, solves,retall=True)
    
        #var = 0
        #for i in range(1,N):
        #    var += norm[0,i]*c[i]**2
        #Var.append(c[0])

        var = cp.Var(U_hat, a)
        Var.append(dt*np.sum(np.abs(V_analytical(x) - var)))
        
    plt.plot(K,Var)

    legend.append("M = %d" % (m))
    
    
plt.ylim(10**-9,10**1)
plt.xlabel("k")
plt.ylabel("Variance")
#plt.title("Orthogonal series expansion of the sign function")
#legend.append("True function")
plt.legend(legend, loc=1)
#print K
plt.yscale('log')
plt.savefig("k_convergence.png")

plt.show()
