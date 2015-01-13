import chaospy as cp
from math import factorial
import pylab as plt
import numpy as np


def E_analytical(x):
    #return (1-np.exp(-10*x))/(10*x)
    #return np.exp(-10*x)*(np.exp(-5*x)-1)/(5*x)
    return 10*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    #return (1-np.exp(-20*x))/(20*x) - ((1-np.exp(-10*x))/(10*x))**2
    return 5*(1-np.exp(-0.2*x))/(x) - (10*(1-np.exp(-0.1*x))/(x))**2


def u(x,a):
    ax = np.outer(a,x)
    return np.exp(-ax)



M = 5
D = 1
N = factorial(M+D)/factorial(M) - 1
a = cp.Uniform(0,0.1)

#V = cp.variable(N)
V = cp.basis(0,M,1)
P_M =  [V[0]]

for n in xrange(1,N):
    summation = 0
    for m in xrange(0,n):
        summation += P_M[m]*cp.E(V[n]*P_M[m],a)/cp.E(P_M[m]**2,a)
    P_M.append(V[n] - summation)
    

x = plt.linspace(-2,2,100)
dt = abs(x[1]-x[0])
m = 1
legend = []
for p in P_M:
    plt.plot(x, p(x),linewidth=2)
    legend.append("M = %d" % m)
    m+=1
    
plt.ylim([-1.5,1.5])
plt.legend(legend, loc=2)
#plt.xlabel("Y")
#plt.ylabel("X")
plt.savefig("gramschmidtpoly.png")
plt.show()


"""
x = plt.linspace(0.00001,10,100)

P_M = cp.Poly(P_M)



P_TTR = cp.orth_ttr(M, a)

M = np.arange(1,M-1)
   
pp = cp.outer(P_M,P_M)
m = cp.E(pp,a)
m = m - cp.diag(cp.diag(m))
error = sum(m)

pp = cp.outer(P_TTR,P_TTR)
m = cp.E(pp,a)
m = m - cp.diag(cp.diag(m))
errorTTR = sum(m)


plt.figure()
plt.plot(error, linewidth=2)
plt.plot(errorTTR, linewidth=2)

#plt.plot(M,var)


plt.xlabel("Terms, M")
plt.ylabel("Error")
#plt.yscale('log')
#plt.title("Error ")
plt.legend(["Gram-Schimdt","TTR"],loc=2)
plt.savefig("gramschmidterror2.png")

plt.show()
"""
