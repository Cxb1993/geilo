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

a = cp.Uniform(0,0.1)
I = cp.Uniform(8,10)
dist = cp.J(a,I)

x = np.linspace(0, 10, 1000)
dt = 10/1000.

M = 12
P = cp.orth_ttr(M, dist)
nodes = dist.sample(10**3)
solves = [u(x, s[0], s[1]) for s in nodes.T]
U_analytic = cp.fit_regression(P, nodes, solves,rule="LS")
#nodes, weights = cp.generate_quadrature(M+1, dist, rule="G")
#solves = [u(x, s[0], s[1]) for s in nodes.T]
#U_analytic = cp.fit_quadrature(P, nodes, weights, solves)


legend = []
pl.rc("figure", figsize=[6,4])
pl.plot(-1,1, "k-")
pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "b")
pl.plot(-1,1, "g")
pl.legend(["Mean","Variance", "Spectral projection","Point collocation","Intrusive Galerkin"],loc=3,prop={"size" :12})
pl.xlim([5,30])
pl.ylim([10**-22,10**0])




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
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))

    #error = cp.E(U_hat,dist)
    #var = cp.Var(U_hat,dist)
    
#pl.plot(x,error,linewidth=2)
#pl.plot(x, var,linewidth=2)

   
    
pl.plot(K,error,"r-",linewidth=2)
pl.plot(K, var,"r--",linewidth=2)


N = 5

#Point collocation
error = []
var = []
K = []
for n in range(1,N):
    P = cp.orth_ttr(n, dist)
    nodes = dist.sample(2*len(P), "M")
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in nodes.T]
    U_hat = cp.fit_regression(P, nodes, solves,rule="T")

    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - cp.E(U_hat,dist))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))


    #error = cp.E(U_hat,dist)
    #var = cp.Var(U_hat,dist)
    
#pl.plot(x,error,linewidth=2)
#pl.plot(x, var,linewidth=2)

   
    
pl.plot(K,error,"b-",linewidth=2)
pl.plot(K, var,"b--",linewidth=2)






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
    sE_ank = cp.sum(E_ank,0)

    def f(c_k,x):
        return -cp.sum(c_k*E_ank,-1)/norm
        
    solver = odespy.RK4(f)
    c_0 = E_ik/norm
    solver.set_initial_condition(c_0)
    c_n, x_ = solver.solve(x)
    #print c_n[:,0]
    U_hat = cp.sum(P*c_n,-1)


    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist) - c_n[:,0])))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist) - cp.Var(U_hat,dist))))

    #error = cp.E(U_hat,dist)
    #var = cp.Var(U_hat,dist)
    
#pl.plot(x,error,linewidth=2)
#pl.plot(x, var,linewidth=2)

    
pl.plot(K,error,"g-",linewidth=2)
pl.plot(K, var,"g--",linewidth=2)

pl.xlabel("Terms, N")
pl.ylabel("Error")
pl.yscale('log')
#pl.title("Error in expectation value and variance ")
#pl.legend(["E, GQ","Var, GQ", "E, PC","Var, PC,", "E, IG","Var, IG"])
pl.savefig("convergence_gallerkin.png")
pl.show()
