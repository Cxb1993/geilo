import chaospy as cp
import numpy as np
import pylab as plt

def u(x,a, I):
    return I*np.exp(-a*x)


C = [[2,1],[1,2]]
mu = np.array([0.5,0.5])
dist_R1 = cp.J(cp.Normal(),cp.Normal())
dist_R2 = cp.J(cp.Uniform(),cp.Uniform())
dist_R3 = cp.J(cp.Beta(4,2),cp.Beta(2,4))

dist_Q = cp.MvNormal(mu,C)

x = np.linspace(0, 1, 100)
dt = x[1]-x[0]
M = 10

s_R1 = dist_R1.sample(10**3,"H")

P = cp.orth_ttr(M, dist_R1)
s_R = dist_R1.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R1.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")


N = 5
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R1)
    s_R = dist_R1.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R1.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="T")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R1) - cp.E(U_hat,dist_R1))/cp.E(U_analytic,dist_R1)))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R1) - cp.Var(U_hat,dist_R1))/cp.Var(U_analytic,dist_R1)))
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R1) - cp.E(U_hat,dist_R1))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R1) - cp.Var(U_hat,dist_R1))))



    
plt.plot(K,error,linewidth=2)
plt.plot(K, var,linewidth=2)





P = cp.orth_ttr(M, dist_R2)
s_R = dist_R2.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R2.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")


N = 5
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R2)
    s_R = dist_R2.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R2.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="T")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R2) - cp.E(U_hat,dist_R2)/cp.E(U_analytic,dist_R2))))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R2) - cp.Var(U_hat,dist_R2))/cp.Var(U_analytic,dist_R2)))
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R2) - cp.E(U_hat,dist_R2))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R2) - cp.Var(U_hat,dist_R2))))



    
plt.plot(K,error,linewidth=2)
plt.plot(K, var,linewidth=2)


P = cp.orth_ttr(M, dist_R3)
s_R = dist_R3.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R3.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")


N = 5
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R3)
    s_R = dist_R3.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R3.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_R, solves,rule="T")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R3) - cp.E(U_hat,dist_R3))/cp.E(U_analytic,dist_R3)))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R3) - cp.Var(U_hat,dist_R3))/cp.Var(U_analytic,dist_R3)))
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R3) - cp.E(U_hat,dist_R3))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R3) - cp.Var(U_hat,dist_R3))))



    
plt.plot(K,error,linewidth=2)
plt.plot(K, var,linewidth=2)















plt.xlabel("Samples, k")
plt.ylabel("Error")
plt.yscale('log')
plt.title("Error in expectation value and variance ")
plt.legend(["E, N","Var, N", "E, U","Var, U", "E, G","Var, G"])
plt.savefig("rosenblatt.png")
plt.show()
