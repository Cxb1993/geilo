import chaospy as cp
import pylab as pl
import numpy as np

cp.seed(1124)
#cp.seed(9125593)
pl.rc("figure", figsize=[6,4])

Nt = 10**2
N = 10

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
legend = []

#pl.plot(-1,1, "k-")
#pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "y")
pl.plot(-1,1, "k")
pl.plot(-1,1, "b")
pl.legend(["(Pseudo-)Random, $[p_{05}, p_{95}]$","Latin Hyperscube, $[p_{05}, p_{95}]$","Sobol","Halton"],loc=3,prop={"size" :12})
pl.rc("figure", figsize=[6,4])
pl.xlim([5,55])
pl.ylim([10**-9,10**0])

N=7

#errorb = np.zeros(Nt)
#varb = np.zeros(Nt)
errorb = []
varb=[]
for i in range(0,100):
    
    error = []
    var = []
    K = []
    for n in range(0,N):
        P = cp.orth_ttr(n, dist)
        nodes = dist.sample(2*len(P))
        K.append(2*len(P))
        solves = [u(T, s[0], s[1]) for s in nodes.T]
        U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

        error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
        var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))

    errorb.append(error)   
    varb.append(var)

errorb = np.array(errorb)
varb = np.array(varb)

p_10 = np.percentile(errorb,5,0)
p_90 = np.percentile(errorb,95,0)
print p_90.shape
print len(K)
meanE = np.sum(errorb,0)/10.
meanVar = np.sum(errorb,0)/10.

pl.fill_between(K,p_10,p_90,facecolor="red",alpha=0.3,lw=0)
#pl.plot(K,meanE,"r-", linewidth=2)
#pl.plot(K, meanVar,"r--", linewidth=2)





errorb = []
varb=[]
for i in range(0,100):
    
    error = []
    var = []
    K = []
    for n in range(0,N):
        P = cp.orth_ttr(n, dist)
        nodes = dist.sample(2*len(P), "LH")
        K.append(2*len(P))
        solves = [u(T, s[0], s[1]) for s in nodes.T]
        U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

        error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
        var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))

    errorb.append(error)   
    varb.append(var)

meanE = np.sum(errorb,0)/10.
meanVar = np.sum(errorb,0)/10.

errorb = np.array(errorb)
varb = np.array(varb)

p_10 = np.percentile(errorb,5,0)
p_90 = np.percentile(errorb,95,0)
print p_90.shape
print len(K)
meanE = np.sum(errorb,0)/10.
meanVar = np.sum(errorb,0)/10.

pl.fill_between(K,p_10,p_90,facecolor="yellow",alpha=0.3,lw=0)


#pl.plot(K,meanE,"b-", linewidth=2)
#pl.plot(K, meanVar,"b--", linewidth=2)





error = []
var = []
K = []
for n in range(0,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "S")
    K.append(2*len(P))
    solves = [u(T, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))


#pl.plot(K,error,"g-", linewidth=2)
pl.plot(K, var,"k-", linewidth=2)



error = []
var = []
K = []
for n in range(0,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "H")
    K.append(2*len(P))
    solves = [u(T, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(E_analytical(T) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(T) - cp.Var(U_hat,dist))))


#pl.plot(K,error,"y-", linewidth=2)
pl.plot(K, var,"b-", linewidth=2)





 

pl.xlabel("Nodes, K")
pl.ylabel("Error")
pl.yscale('log')
#pl.title("Error in expectation value and variance ")
#pl.legend(["Expectation value","Variance", "Expectation value, Hammersley","Variance, Hammersley"])
pl.savefig("convergence_collocation_compare.png")

pl.show()

pl.figure()
nodes = dist.sample(100)
pl.scatter(nodes[0],nodes[1])
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("samples.png")

pl.figure()
nodes = dist.sample(100, "H")
pl.scatter(nodes[0],nodes[1])
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("samples_H.png")

pl.figure()
nodes = dist.sample(100, "LH")
pl.scatter(nodes[0],nodes[1])
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("samples_L.png")

pl.figure()
nodes = dist.sample(100, "S")
pl.scatter(nodes[0],nodes[1])
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("samples_S.png")


#pl.show()




