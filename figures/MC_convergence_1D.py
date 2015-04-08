import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2


legend = []
pl.rc("figure", figsize=[6,4])
#pl.plot(-1,1, "k-")
#pl.plot(-1,1, "k--")
#pl.plot(-1,1, "r")
#pl.plot(-1,1, "b")
#pl.plot(-1,1, "g")
#pl.legend(["Mean","Variance", "Monte Carlo","Polynomial chaos"],loc=3,prop={"size" :12})

pl.xlim([0,20])
#pl.ylim([10**-26,10**1])

def E_analytical(x):
    #return (1-np.exp(10*x))/(10*x)
    #return np.exp(-10*x)*(np.exp(-5*x)-1)/(5*x)
    out = 10*(1-np.exp(-0.1*x))/(x)
    out[x==0] = 11
    return out

    
def V_analytical(x):
    #return (1-np.exp(-20*x))/(20*x) - ((1-np.exp(-10*x))/(10*x))**2
    out = 5*(1-np.exp(-0.2*x))/(x) - (10*(1-np.exp(-0.1*x))/(x))**2
    out[x==0] = 21./20 - 11**2
    return out

def u(x,a):
    ax = np.outer(a,x)
    return np.exp(-ax)

 
a = cp.Uniform(0, 0.1)
x = np.linspace(0, 10, Nt)[1:]
dt = 10./(Nt-1)






N = 22
   
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

K = []
N = 15

for k in range(0,N+1):
    P, norm = cp.orth_ttr(k, a, retall=True)
    N = len(P)
    nodes, weights = cp.generate_quadrature(2*k, a, rule="E")
    K.append(len(nodes[0]))
    
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




pl.plot(totalerrorMC, "r-",linewidth=2)
pl.plot(totalvarianceMC, "r--", linewidth=2)

#pl.plot(K,errorPoly, "b-", linewidth=2)
#pl.plot(K,varPoly,"b--", linewidth=2)

#pl.plot(K,errorCP,"g-", linewidth=2)
#pl.plot(K,varCP,"g--", linewidth=2)

pl.xlabel("Model evaluations")
pl.ylabel("Error")
pl.yscale('log')
pl.legend(["Mean","Variance", "Polynomial chaos, mean","Polynomial chaos, variance"],loc=3,prop={"size" :12})
#pl.legend(["E, MC","Var, MC","E, PC","Var, PC"], loc=3)
#pl.savefig("Convergence_repeat.svg")
pl.savefig("MC_convergence_1D_1.svg")

pl.show()
