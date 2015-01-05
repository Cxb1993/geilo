import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**3
N = 10

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(t,a, I):
    return I*np.exp(-a*t)

 
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)
T = np.linspace(0.000000001, 10, Nt)
dt = T[1] - T[0]


    
def MC():
    samples_a = a.sample(N)
    samples_I = I.sample(N)
    
    U = [u(T,q,i) for q,i in zip(samples_a.T, samples_I.T)]
    U = np.array(U)

    E = (np.cumsum(U, 0).T/np.arange(1,N+1)).T
    V = (np.cumsum((U-E)**2, 0).T/np.arange(1,N+1)).T
    #V = (np.cumsum((U**2), 0).T/np.arange(1,N+1)).T - E**2
    
    
    error = []
    var = []
    for n in xrange(N):
        error.append(dt*np.sum(np.abs(E_analytical(T) - E[n,:])))
        var.append(dt*np.sum(np.abs(V_analytical(T) - V[n,:])))
        
    return np.array(error), np.array(var)


reruns = 10**2
totalerrorMC = np.zeros(N)
totalvarianceMC = np.zeros(N)
for i in xrange(reruns):
    errorMC,varianceMC = MC()

    totalerrorMC = np.add(totalerrorMC, errorMC)
    totalvarianceMC = np.add(totalvarianceMC, varianceMC)
    
       
totalerrorMC = np.divide(totalerrorMC, reruns)
totalvarianceMC = np.divide(totalvarianceMC, reruns)


errorCP = []
varCP = []
a_m = cp.E(a)
I_m = cp.E(I)
for n in xrange(1,N):
    print n
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n, dist, rule="G")
    solves = u(nodes[0],a_m,I_m)
    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    print cp.Var(U_hat,dist)
    print np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist)))
    #errorCP.append(dt*np.sum(np.abs(cp.E(u,dist) - cp.E(U_hat,dist))))
    #varCP.append(dt*np.sum(np.abs(cp.Var(u,dist) - cp.Var(U_hat,dist))))
    #errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    #varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))
    errorCP.append(dt*np.sum(np.abs(E_analytical(T))) - cp.E(U_hat,dist))
    varCP.append(dt*np.sum(np.abs(V_analytical(T))) - cp.Var(U_hat,dist))




pl.plot(totalerrorMC[:])
pl.plot(totalvarianceMC[:])
pl.plot(errorCP)
pl.plot(varCP)
pl.xlabel("Terms, N")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["Expectation value MC","Variance MC","Expectation value CP","Variance CP"])
pl.savefig("MC_convergence_2D.png")

pl.show()
