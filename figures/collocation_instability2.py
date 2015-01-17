import chaospy as cp
import numpy as np
import pylab as plt


Nt = 10**2
N = 10

"""
plt.plot(-1,1, "k-")
plt.plot(-1,1, "k--")
plt.plot(-1,1, "r")
plt.plot(-1,1, "b")
#pl.plot(-1,1, "g")
plt.legend(["E","Var","Least square", "Tikhanov"],loc=3,prop={"size" :12})
plt.xlim([1,17])
"""

def u(x,a, I):
    return I*np.exp(-a*x)

#def u(x,a, I):
#    return I*np.exp(-a*x)




    
M = 8
k=20
    
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
b = cp.Normal()

dist = cp.J(a,I)

x = np.linspace(0, 1, Nt+1)[1:]
dt = 1./Nt

K = range(2,20)
error = []
var=[]

plt.figure()
plt.yscale('log')

#for k in K:
P = cp.orth_ttr(M, dist)
nodes = dist.sample(k, "M")
solves = [u(x, *s) for s in nodes.T]
U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

plt.plot(x,cp.E(U_hat,dist),"r-",linewidth=2)
plt.plot(x, cp.Var(U_hat,dist),"r--",linewidth=2)

error = []
var=[]

plt.figure()
plt.yscale('log')

#for k in K:
P = cp.orth_ttr(M, dist)
nodes = dist.sample(k, "M")
solves = [u(x, *s) for s in nodes.T]
U_hat = cp.fit_regression(P, nodes, solves,rule="T")

plt.plot(x,cp.E(U_hat,dist),"b-",linewidth=2)
plt.plot(x, cp.Var(U_hat,dist),"b--",linewidth=2)
    
    
    




plt.xlabel("Samples, K")
plt.ylabel("Error")
plt.yscale('log')
#plt.xlim([0,50])
#plt.title("Error in expectation value and variance ")
#plt.legend(["E","Var"])
#plt.savefig("convergence_LSvsT.png")
plt.show()
