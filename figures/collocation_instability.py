import chaospy as cp
import numpy as np
import pylab as plt


Nt = 10**2
N = 10


plt.plot(-1,1, "k-")
plt.plot(-1,1, "k--")
plt.plot(-1,1, "r")
plt.plot(-1,1, "b")
#pl.plot(-1,1, "g")
plt.legend(["E","Var","Least square", "Tikhanov"],loc=3,prop={"size" :12})
plt.xlim([1,17])

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)

#def u(x,a, I):
#    return I*np.exp(-a*x)




    
M = 4

    
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
b = cp.Normal()

dist = cp.J(a,I)

x = np.linspace(0, 10, Nt+1)[1:]
dt = 10./Nt


P = cp.orth_ttr(8, dist)
#nodes, weights = cp.generate_quadrature(8, dist, rule="E")
#solves = [u(x, *s) for s in nodes.T]
#U_analytic = cp.fit_quadrature(P, nodes, weights, solves)
samples = dist.sample(10**3)
solves = [u(x, *s) for s in samples.T]
U_analytic = cp.fit_regression(P, samples, solves,rule="T")






K = range(2,20)
error = []
var=[]
for k in K:
    P = cp.orth_ttr(M, dist)
    nodes = dist.sample(k, "M")
    solves = [u(x, *s) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))

plt.plot(K,error,"r-",linewidth=2)
plt.plot(K, var,"r--",linewidth=2)

error = []
var=[]

for k in K:
    P = cp.orth_ttr(M, dist)
    nodes = dist.sample(k, "M")
    solves = [u(x, *s) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="T")

    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))

plt.plot(K,error,"b-",linewidth=2)
plt.plot(K, var,"b--",linewidth=2)





plt.xlabel("Nodes, K")
plt.ylabel("Error")
plt.yscale('log')
plt.xlim([0,50])
#plt.title("Error in expectation value and variance ")
#plt.legend(["E","Var"])
plt.savefig("convergence_LSvsT.png")
plt.show()
