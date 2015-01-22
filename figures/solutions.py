import chaospy as cp
import pylab as plt
import numpy as np
import odespy

def plot(t,E,Var,name):
    plt.rc("figure", figsize=[6,4])

    plt.figure()
    plt.plot(t,E,linewidth=2)
    plt.plot(t,Var,linewidth=2)
    plt.xlabel("Time, t")
    plt.ylabel("Distance, s")
    plt.legend(["Mean","Variance"], loc=2)
    plt.savefig(name+".png")
    plt.show()


    
def s(t, v):
    return v*t

v = cp.Normal(5,1)
t = np.linspace(0,10,1000)
M = 5

P = cp.orth_ttr(M, v)
nodes, weights = cp.generate_quadrature(M+1, v, rule="G")
solves = [s(t, n) for n in nodes.T]
U_hat = cp.fit_quadrature(P, nodes, weights, solves)

E = cp.E(U_hat,v)
Var = cp.Var(U_hat,v)


#plot(t,E,Var, "solution1")






###
# Set 2 of excercises
###

def s(t, v0, a):
    return v0*t + 0.5*a*t**2


N = 1000
v0 = cp.Uniform(1,2)
a = cp.Beta(2,2)
t = np.linspace(0,10,1000)

samples_v0 = v0.sample(N)
samples_a = a.sample(N)

distance = np.array([s(t,v0_,a_) for v0_,a_ in zip(samples_v0.T, samples_a.T)])
E = np.sum(distance,0)/N
Var = np.sum(distance**2,0)/N - E**2

#plot(t,E,Var)
#plot(t,E,Var, "solution2")


def s(t, v0, a):
    return v0*t + 0.5*a*t**2

N = 1000
v0 = cp.Uniform(1,2)
a = cp.Beta(2,2)
t = np.linspace(0,10,1000)

samples_v0 = v0.sample(N, "S")
samples_a = a.sample(N, "S")

distance = np.array([s(t,v0_,a_) for v0_,a_ in zip(samples_v0.T, samples_a.T)])
E = np.sum(distance,0)/N
Var = np.sum(distance**2,0)/N - E**2




#plot(t,E,Var)
#plot(t,E,Var, "solution3")





def s(t, v0, a):
    return v0*t + 0.5*a*t**2

v0 = cp.Uniform(1,2)
a = cp.Beta(2,0.5)
dist = cp.J(v0,a)
t = np.linspace(0,10,1000)

M = 5

P = cp.orth_ttr(M, dist)
nodes, weights = cp.generate_quadrature(M+1, dist, rule="G")
solves = [s(t, *n) for n in nodes.T]
U_hat = cp.fit_quadrature(P, nodes, weights, solves)

E = cp.E(U_hat, dist)
Var = cp.Var(U_hat, dist)

#plot(t,E,Var)
#plot(t,E,Var, "solution4")




def s(t, v0, a):
    return v0*t + 0.5*a*t**2

v0 = cp.Uniform(1,2)
a = cp.Beta(2,2)
dist = cp.J(v0,a)
t = np.linspace(0,10,1000)

M = 5

P = cp.orth_ttr(M, dist)
nodes, weights = cp.generate_quadrature(M+1, dist, rule="C", sparse=True)
solves = [s(t, *n) for n in nodes.T]
U_hat = cp.fit_quadrature(P, nodes, weights, solves)

E = cp.E(U_hat, dist)
Var = cp.Var(U_hat, dist)

#plot(t,E,Var)
#plot(t,E,Var, "solution5")





def s(t, v0, a):
    return v0*t + 0.5*a*t**2

v0 = cp.Uniform(1,2)
a = cp.Beta(2,2)
dist = cp.J(v0,a)

t = np.linspace(0,10,1000)
M = 5

P = cp.orth_ttr(M, dist)
nodes = dist.sample(2*len(P))
solves = [s(t, *n) for n in nodes.T]
U_hat = cp.fit_regression(P, nodes, solves,rule="LS")

E = cp.E(U_hat, dist)
Var = cp.Var(U_hat, dist)

#plot(t,E,Var)
#plot(t,E,Var, "solution6")




def s(t, v0, a):
    return v0*t + 0.5*a*t**2

v0 = cp.Uniform(1,2)
a = cp.Beta(2,2)
dist = cp.J(v0,a)

t = np.linspace(0,10,1000)
M = 5

P = cp.orth_ttr(M, dist)
nodes = dist.sample(2*len(P), "M")
solves = [s(t, *n) for n in nodes.T]
U_hat = cp.fit_regression(P, nodes, solves,rule="T")

E = cp.E(U_hat, dist)
Var = cp.Var(U_hat, dist)

#plot(t,E,Var)
#plot(t,E,Var, "solution7")




#Problems day 3

def plot(t,E,Var,name):
    plt.rc("figure", figsize=[6,4])

    plt.figure()
    plt.plot(t,E,linewidth=2)
    plt.plot(t,Var,linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.legend(["Mean","Variance"], loc=2)
    plt.savefig(name+".png")
    plt.show()




a = cp.Normal(4,1)
I = cp.Uniform(2, 6)
dist = cp.J(a, I)

x = np.linspace(0,1,100)


P, norm = cp.orth_ttr(5, dist, retall=True)

q0, q1 = cp.variable(2)

P_nk = cp.outer(P, P)
E_ak = cp.E(q0*P, dist)
E_ik = cp.E(q1*P, dist)
E_nk = cp.E(P_nk, dist)

def f(c_k,x):
    return (c_k + E_ak)*cp.sum(E_nk, -1)/norm
        
solver = odespy.RK4(f)
c_0 = E_ik/norm
solver.set_initial_condition(c_0)
c_n, x_ = solver.solve(x)
#print c_n[:,0]
U_hat = cp.sum(P*c_n,-1)

E = cp.E(U_hat, dist)
Var = cp.Var(U_hat, dist)
plot(x,E,Var, "solution8")



# MC test
x = np.linspace(0,1,100)
def u(x,a,I):
    return a*(np.exp(x)-1) + I*np.exp(x)
    
N = 1000
s_a = a.sample(N)
s_I = I.sample(N)


distance = np.array([u(x,a_,I_) for a_,I_ in zip(s_a.T, s_I.T)])
E = np.sum(distance,0)/N
Var = np.sum(distance**2,0)/N - E**2

plot(x,E,Var, "solution8mc")






a = cp.Normal(4,1)
I = cp.Uniform(2, 6)
dist_Q = cp.J(a, I)
dist_R = cp.J(cp.Normal(), cp.Uniform())

x = np.linspace(0,1,100)

P = cp.orth_ttr(2, dist_R)
nodes_R, weights_R = cp.generate_quadrature(3, dist_R)
nodes_Q = dist_Q.inv(dist_R.fwd(nodes_R))
weights_Q = weights_R*dist_Q.pdf(nodes_Q)/dist_R.pdf(nodes_R)

samples_u = [u(x, *node) for node in nodes_Q.T]
u_hat = cp.fit_quadrature(P, nodes_R, weights_Q, samples_u)

plot(x,E,Var, "solution9")
