import chaospy as cp
import pylab as pl
import numpy as np

Nt = 10**3
N = 100


def E_analytical(x):
    #return (1-np.exp(-10*x))/(10*x)
    #return np.exp(-10*x)*(np.exp(-5*x)-1)/(5*x)
    return 10*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    #return (1-np.exp(-20*x))/(20*x) - ((1-np.exp(-10*x))/(10*x))**2
    return 5*(1-np.exp(-0.2*x))/(x) - (10*(1-np.exp(-0.1*x))/(x))**2
    
def run():
    def u(t,a):
        return np.exp(-a*t)

 
    a = cp.Uniform(0, 0.1)

    samples = a.sample(N)

    T = np.linspace(0.000000001, 10, Nt)
    dt = T[1] - T[0]

    U = [u(T,q) for q in samples.T]
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
totalerror = np.zeros(N)
totalvariance = np.zeros(N)
for i in xrange(reruns):
    error,variance = run()
    totalerror = np.add(totalerror, error)
    totalvariance = np.add(totalvariance, variance)
    
totalerror = np.divide(totalerror, reruns)
totalvariance = np.divide(totalvariance, reruns)

pl.plot(totalerror)
pl.plot(totalvariance)
pl.xlabel("Terms, N")
pl.ylabel("Error")
 pl.yscale('log')
pl.title("Error Expectation value and Variance for Monte Carlo Sampling")
pl.legend(["Expectation value MC","Variance MC"])
pl.savefig("MC_convergence_1D.png")

pl.show()
