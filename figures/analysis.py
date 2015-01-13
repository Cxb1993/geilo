import chaospy as cp
import pylab as plt
import numpy as np

def u(x,a, I):
    return I*np.exp(-a*x)

a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)

x = np.linspace(0, 10, 100)
M = 5

P, norm = cp.orth_ttr(M, dist, retall=True)
nodes, weights = cp.generate_quadrature(M+1, dist, rule="G")
solves = [u(x, s[0], s[1]) for s in nodes.T]
U_hat, c = cp.fit_quadrature(P, nodes, weights, solves, retall=True)


s = dist.sample(10**5)
u_mc = U_hat(*s)
mean = np.mean(u_mc,1)
var = np.var(u_mc,1)


p_10 = np.percentile(u_mc,10,1)
p_90 = np.percentile(u_mc,90,1)


plt.figure()
plt.plot(x, cp.E(U_hat,dist), linewidth=2)
plt.plot(x[::10], c[0,::10], "o",linewidth=2)
plt.plot(x[5::10], mean[5::10], "o",linewidth=2)
plt.xlabel("x")
plt.ylabel("E")
plt.legend(["cp.E(U_hat,dist)","c[0]","np.mean(u_mc,0)"])
plt.savefig("E.png")


plt.figure()
plt.plot(x, cp.Var(U_hat,dist),linewidth=2)
plt.plot(x[::10], np.sum(norm[1:]*c[1:].T**2,1)[::10], "o",linewidth=2)
plt.plot(x[5::10], var[5::10], "o",linewidth=2)
plt.xlabel("x")
plt.ylabel("Var")
plt.legend(["cp.Var(U_hat,dist)","np.sum(norm[1:]*c[1:].T**2,1)","np.var(u_mc,0)"])
plt.savefig("Var.png")
#plt.show()


plt.figure()
plt.plot(p_10,linewidth=2)
plt.plot(p_90,linewidth=2)
plt.plot(cp.E(U_hat,dist),linewidth=2)
plt.xlabel("x")
plt.ylabel("u")
plt.legend(["$P_{10}$", "$P_{90}$", "E"])
plt.savefig("percentiles.png")
#plt.show()

q_0,q_1 = cp.variable(2)
cp.E_cond(U_hat**2,q_0,dist)

S_Ti = cp.Sens_t(U_hat, dist)

plt.figure()
plt.plot(x,S_Ti[0],linewidth=2)
plt.plot(x,S_Ti[1],linewidth=2)
plt.xlabel("x")
plt.ylabel("Sensitivity")
plt.legend(["a","I"])
#plt.axis([0,10,-6*10**-15,6*10**-15])
plt.savefig("sens.png")
plt.show()

