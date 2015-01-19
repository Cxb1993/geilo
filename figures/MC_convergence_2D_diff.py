import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**2
N = 100


legend = []
pl.rc("figure", figsize=[6,4])
pl.plot(-1,1, "k-")
pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "b")
pl.plot(-1,1, "g")
pl.legend(["Mean","Variance", "Monte Carlo","Gaussian Quadrature, full tensor grid", "Least Square with Sobol sampling"],loc=3,prop={"size" :12})
#pl.xlim([0,20])
pl.ylim([10**-28,10**4])


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
ptotalvarianceMC = np.divide(totalvarianceMC, reruns)


errorCP = []
varCP = []

K = []

N = 7
for n in xrange(0,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(2**n+1, dist, rule="G")
    K.append(len(nodes[0]))
    i1,i2 = np.mgrid[:len(weights), :Nt]
    solves = u(T[i2],nodes[0][i1],nodes[1][i1])

    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    errorCP.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    varCP.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))

pl.plot(K,errorCP,"b-",linewidth=2)
pl.plot(K, varCP,"b--",linewidth=2)

N=9
error = []
var=[]
K = []
for k in xrange(0,N+1):
    P = cp.orth_ttr(k, dist)
    nodes = dist.sample(2*len(P), "S")
    K.append(2*len(P))
    solves = [u(T, *s) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))    


pl.plot(K,error,"g-",linewidth=2)
pl.plot(K, var,"g--",linewidth=2)


    
pl.rc("figure", figsize=[6,4])

pl.plot(totalerrorMC[:],"r-",linewidth=2)
pl.plot(totalvarianceMC[:],"r--",linewidth=2)
pl.xlabel("Nodes, K")
pl.ylabel("Error")
pl.xlim([0,100])
pl.yscale('log')
#pl.legend(["E, MC","Var, MC","E, PC","Var, PC"],loc=3)
pl.savefig("MC_convergence_2D_diff.png")

pl.show()
