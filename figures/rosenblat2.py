import chaospy as cp
import numpy as np
import pylab as plt

def u(x,a, I):
    return I*np.exp(-a*x)













legend = []
plt.rc("figure", figsize=[6,4])
plt.plot(-1,1, "k-")
plt.plot(-1,1, "k--")
plt.plot(-1,1, "r")
plt.plot(-1,1, "b")
plt.plot(-1,1, "g")
plt.legend(["mean","Variance", "Normal","Uniform","Gamma"],loc=3,prop={"size" :12})
#plt.xlim([5,90])
#plt.ylim([5*10**-5,10**1])


def u(x,a, I):
    return I*np.exp(-a*x)


x = np.linspace(0, 10, 1000)
dt = 10/1000.

#P = cp.orth_ch

dist_R = cp.J(cp.Uniform(-1,1),cp.Uniform(-1,1))

dist_Q = cp.J(cp.Uniform(0.1),cp.Uniform(8,10))

M = 12

P = cp.orth_ttr(M, dist_R)
s_R = dist_R.sample(2*len(P), "M")
s_Q = dist_Q.inv(dist_R.fwd(s_R))

solves = [u(x, s[0], s[1]) for s in s_Q.T]
U_analytic = cp.fit_regression(P, s_R, solves,rule="LS")


N = 8
error = []
var = []
K = []
for n in range(1,N+1):
    P = cp.orth_ttr(n, dist_R)
    s_R = dist_R.sample(2*len(P), "M")
    s_Q = dist_Q.inv(dist_R.fwd(s_R))
    K.append(2*len(P))
    solves = [u(x, s[0], s[1]) for s in s_Q.T]
    U_hat = cp.fit_regression(P, s_Q, solves,rule="LS")
    #error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_R3) - cp.E(U_hat,dist_R3))/cp.E(U_analytic,dist_R3)))
    #var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_R3) - cp.Var(U_hat,dist_R3))/cp.Var(U_analytic,dist_R3)))
    error.append(dt*np.sum(np.abs(cp.E(U_analytic,dist_Q) - cp.E(U_hat,dist_R))))
    var.append(dt*np.sum(np.abs(cp.Var(U_analytic,dist_Q) - cp.Var(U_hat,dist_R))))



    
plt.plot(K,error,"g-",linewidth=2)
plt.plot(K, var,"g--",linewidth=2)










plt.xlabel("Nodes, K")
plt.ylabel("Error")
plt.yscale('log')
#plt.title("Error in expectation value and variance ")
#plt.legend(["E, N","Var, N", "E, U","Var, U", "E, G","Var, G"],loc=3)
#plt.savefig("rosenblatt.png")
plt.show()
