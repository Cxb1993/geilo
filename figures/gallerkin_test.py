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
x = np.linspace(0, 3, 300)[1:]
dt = x[1] - x[0]

M = 4
D = 2

#Intrusive Gallerkin
N= 7
error = []
var = []
K = []
for n in range(1,N):

    P, norm = cp.orth_ttr(n, dist, retall=True, normed=True)
    q0, q1 = cp.variable(2)
    K.append(len(P))

    P_nk = cp.outer(P, P)
    E_ank = cp.E(q0*P_nk, dist)
    E_ik = cp.E(q1*P, dist)

    E_nn = cp.E(P_nk,dist)

    def f(c_n,x):
        out = -c_n/norm*cp.sum(E_ank,0)
        if np.isinf(out).any():
            print norm
            print cp.sum(E_ank,0)
            print c_n, x
            assert False
        return out

    solver = odespy.Euler(f)
    c_0 = E_ik/norm
    solver.set_initial_condition(c_0)
    c_n, x_ = solver.solve(x)
    U_hat = cp.sum(P*c_n,-1)

    print cp.Var(U_hat,dist)
    #exit()
    #print np.abs(E_analytical(x) - cp.E(U_hat,dist))
    error.append(cp.sum(cp.E(U_hat,dist)))
    var.append(cp.sum(cp.Var(U_hat,dist)))

    
pl.plot(K,error,linewidth=2)
pl.plot(K, var,linewidth=2)


pl.xlabel("Samples, k")
pl.ylabel("Error")
pl.yscale('log')
pl.title("Error in expectation value and variance ")
pl.legend(["E, GQ","Var, GQ", "E, PC","Var, PC,", "E, IG","Var, IG"])
pl.savefig("convergence_gallerkin.png")
pl.show()
