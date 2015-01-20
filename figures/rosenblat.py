import chaospy as cp
import numpy as np
import pylab as plt

def u(x,a, I):
    return I*np.exp(-a*x)






def E_analytical(x):
    return 0.5*(-x*np.exp(0.5*x**2))






legend = []
plt.rc("figure", figsize=[6,4])
plt.plot(-1,1, "k-")
plt.plot(-1,1, "k--")
plt.plot(-1,1, "r")
plt.plot(-1,1, "b")
plt.plot(-1,1, "g")
plt.legend(["mean","Variance", "Normal","Uniform","Gamma"],loc=3,prop={"size" :12})
plt.xlim([6,55])
plt.ylim([5*10**-5,10**1])




    

C = [[1,0.5],[0.5,1]]
mu = np.array([0,0])
dist_R1 = cp.J(cp.Normal(),cp.Normal())
dist_R2 = cp.J(cp.Uniform(),cp.Uniform())
dist_R3 = cp.J(cp.Beta(4,2),cp.Beta(2,4))

dist_Q = cp.MvNormal(mu,C)

x = np.linspace(0, 1, 1000)
dt = x[1]-x[0]

E0 = []
V0 = []
for i in xrange(0,5):
    samples = dist_Q.sample(10**5, "S")
    solves = [u(_, *samples) for _ in x]
    print i

    E0.append(np.mean(solves, -1))
    V0.append(np.var(solves, -1))

E0 = np.mean(E0,0)
V0 = np.mean(V0,0)
V0 = (V0 - E0**2)*5*10**5/(5.*10**5-1) 

M = 12
"""
s_R1 = dist_R1.sample(10**3,"H")

P = cp.orth_ttr(M, dist_R1)
s_R = dist_R1.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R1.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")
"""
P = cp.orth_ttr(M, dist_R1)
    
nodes_R, weights_R = cp.generate_quadrature(M+1, dist_R1, rule="G")
nodes_Q = dist_Q.inv(dist_R1.fwd(nodes_R))

weights_Q = weights_R*dist_Q.pdf(nodes_Q)/dist_R1.pdf(nodes_R)
    
solves = [u(x, s[0], s[1]) for s in nodes_Q.T]
U_analytic = cp.fit_quadrature(P, nodes_R, weights_Q, solves)


N = 8
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R1)
    s_R = dist_R1.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R1.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="LS")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R1) - cp.E(U_hat,dist_R1))/cp.E(U_analytic,dist_R1)))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R1) - cp.Var(U_hat,dist_R1))/cp.Var(U_analytic,dist_R1)))
    error.append(dt*np.sum(np.abs(E0 - cp.E(U_hat,dist_R1))))
    var.append(dt*np.sum(np.abs(V0 - cp.Var(U_hat,dist_R1))))



    
plt.plot(K,error,"r-",linewidth=2)
plt.plot(K, var,"r--",linewidth=2)




"""

P = cp.orth_ttr(M, dist_R2)
s_R = dist_R2.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R2.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")
"""
nodes_R, weights_R = cp.generate_quadrature(M+1, dist_R2, rule="G")
nodes_Q = dist_Q.inv(dist_R2.fwd(nodes_R))

weights_Q = weights_R*dist_Q.pdf(nodes_Q)/dist_R2.pdf(nodes_R)
    
solves = [u(x, s[0], s[1]) for s in nodes_Q.T]
U_analytic = cp.fit_quadrature(P, nodes_R, weights_Q, solves)



N = 8
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R2)
    s_R = dist_R2.sample(2*len(P)+1, "M")
    s_Q = dist_Q.inv(dist_R2.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="LS")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R2) - cp.E(U_hat,dist_R2)/cp.E(U_analytic,dist_R2))))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R2) - cp.Var(U_hat,dist_R2))/cp.Var(U_analytic,dist_R2)))
    error.append(dt*np.sum(np.abs(E0 - cp.E(U_hat,dist_R2))))
    var.append(dt*np.sum(np.abs(V0 - cp.Var(U_hat,dist_R2))))



    
plt.plot(K,error,"b-",linewidth=2)
plt.plot(K, var,"b--",linewidth=2)

"""

P = cp.orth_ttr(M, dist_R3)
s_R = dist_R3.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R3.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")
"""

P = cp.orth_ttr(M, dist_R3)
    
nodes_R, weights_R = cp.generate_quadrature(M+1, dist_R3, rule="G")
nodes_Q = dist_Q.inv(dist_R3.fwd(nodes_R))

weights_Q = weights_R*dist_Q.pdf(nodes_Q)/dist_R3.pdf(nodes_R)
    
solves = [u(x, s[0], s[1]) for s in nodes_Q.T]
U_analytic = cp.fit_quadrature(P, nodes_R, weights_Q, solves)

N = 8
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R3)
    s_R = dist_R3.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R3.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="LS")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R3) - cp.E(U_hat,dist_R3))/cp.E(U_analytic,dist_R3)))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R3) - cp.Var(U_hat,dist_R3))/cp.Var(U_analytic,dist_R3)))
    error.append(dt*np.sum(np.abs(E0 - cp.E(U_hat,dist_R3))))
    var.append(dt*np.sum(np.abs(V0 - cp.Var(U_hat,dist_R3))))



    
plt.plot(K,error,"g-",linewidth=2)
plt.plot(K, var,"g--",linewidth=2)






#P = cp.orth_ch







plt.xlabel("Nodes, K")
plt.ylabel("Error")
plt.yscale('log')
#plt.title("Error in expectation value and variance ")
#plt.legend(["E, N","Var, N", "E, U","Var, U", "E, G","Var, G"],loc=3)
plt.savefig("rosenblatt2.png")
plt.show()
