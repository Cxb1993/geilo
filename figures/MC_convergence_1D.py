import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2
N = 30

def E_analytical(x):
    #return (1-np.exp(-10*x))/(10*x)
    #return np.exp(-10*x)*(np.exp(-5*x)-1)/(5*x)
    return 10*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    #return (1-np.exp(-20*x))/(20*x) - ((1-np.exp(-10*x))/(10*x))**2
    return 5*(1-np.exp(-0.2*x))/(x) - (10*(1-np.exp(-0.1*x))/(x))**2


def u(x,a):
    ax = np.outer(a,x)
    return np.exp(-ax)

 
a = cp.Uniform(0, 0.1)
x = np.linspace(0.0000000001, 10, Nt)
dt = x[1] - x[0]


    
def MC():
    samples_a = a.sample(N)

    
    U = [u(x,q) for q in samples_a.T]
    U = np.array(U)

    E = (np.cumsum(U, 0).T/np.arange(1,N+1)).T
    V = (np.cumsum((U-E)**2, 0).T/np.arange(1,N+1)).T
    #V = (np.cumsum((U**2), 0).T/np.arange(1,N+1)).T - E**2
    
    
    error = []
    var = []
    for n in xrange(N):
        error.append(dt*np.sum(np.abs(E_analytical(x) - E[n,:])))
        var.append(dt*np.sum(np.abs(V_analytical(x) - V[n,:])))
        
    return np.array(error), np.array(var)


reruns = 10**3
totalerrorMC = np.zeros(N)
totalvarianceMC = np.zeros(N)
for i in xrange(reruns):
    errorMC,varianceMC = MC()

    totalerrorMC = np.add(totalerrorMC, errorMC)
    totalvarianceMC = np.add(totalvarianceMC, varianceMC)
    
       
totalerrorMC = np.divide(totalerrorMC, reruns)
totalvarianceMC = np.divide(totalvarianceMC, reruns)

errorPoly = []
varPoly = []

errorCP = []
varCP = []
m = 2

K = range(m,N+1)

for k in K:
    P, norm = cp.orth_ttr(m, a, retall=True)
    N = len(P)
    nodes, weights = cp.generate_quadrature(k, a, rule="G")
    solves = u(x,nodes[0])
    U_hat, c = cp.fit_quadrature(P, nodes, weights, solves,retall=True)
    
    errorCP.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,a))))
    varCP.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,a))))

    #P = cp.basis(0,m,1)
    #U_hat = cp.fit_regression(P, nodes, solves, rule="T", order=1)

    q = cp.variable()
    P = []
  #  nodei, nodej = np.meshgrid(nodes,nodes)
    for xi in nodes[0]:
        s = q**0
        for xj in nodes[0]:
            if xi != xj:
                s *= (q*1/(xi-xj)) - xj/(xi-xj)
        P.append(s)

    c = u(x, nodes[0])
    #print c

    U_hat = 0
    i = 0
    for p in P:
        U_hat += p*c[i]
        i += 1

    
        
    errorPoly.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,a))))
    varPoly.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,a))))


pl.plot(totalerrorMC)
pl.plot(totalvarianceMC)

pl.plot(K,errorPoly)
pl.plot(K,varPoly)

pl.plot(K,errorCP)
pl.plot(K,varCP)


pl.xlabel("Samples, k")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["Expectation value MC","Variance MC","Expectation value Poly","Variance Poly","Expectation value CP","Variance CP"])
pl.savefig("MC_convergence_1D_3.png")

pl.show()
