import chaospy as cp
import pylab as pl
import numpy as np
from scipy.integrate import ode 
from math import factorial
import odespy

def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)

a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)

x = np.linspace(0, 10, 11)[1:]
dt = x[1] - x[0]

M = 4
D = 2


N = 4

#gaussian quadrature
error = []
var = []
K = []
for n in xrange(1,N+1):
    P = cp.orth_ttr(n, dist)
    nodes, weights = cp.generate_quadrature(n+1, dist, rule="G")
    K.append(len(nodes[0]))
    solves = [u(x, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_quadrature(P, nodes, weights, solves)
    error.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,dist))))
    
pl.figure()
pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)


N = 6

#Point collocation
error = []
var = []
K = []
for n in range(1,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "M")
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

    error.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,dist))))


pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)







#Intrusive Gallerkin
N= 6
error = []
var = []
K = []
for n in range(1,N):

    P, norm = cp.orth_ttr(n, dist, retall=True)

    q0, q1 = cp.variable(2)
    K.append(len(P))

    P_nk = cp.outer(P, P)
    E_ank = cp.E(q0*P_nk, dist)
    E_nn = cp.E(P_nk, dist)
    E_ik = cp.E(q1*P, dist)

    def f(c_n,x):
        return -c_n/norm*np.sum(E_ank,0)

    solver = odespy.RK4(f)
    c_0 = sum(E_ik,0)/norm
    print c_0.shape
    solver.set_initial_condition(c_0)
    c_n, x_ = solver.solve(x)
    print c_n.shape
    U_hat = cp.sum(P*c_n,-1)

    #print  cp.E(U_hat,dist)
    #print  E_analytical(x)
    #print cp.E(U_hat,dist) - E_analytical(x)
    #exit()
    error.append(dt*np.sum(np.abs(E_analytical(x) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(V_analytical(x) - cp.Var(U_hat,dist))))

print error
print var
pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)


print K
pl.xlabel("Samples, k")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["E, GQ","Var, GQ", "E, PC","Var, PC,", "E, IG","Var, IG"])
pl.savefig("convergence_gallerkin.png")
pl.show()
