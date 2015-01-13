import chaospy as cp
import pylab as pl
import numpy as np
from matplotlib import gridspec


def E_analytical(x):
    return 90*(1-np.exp(-0.1*x))/(x)

    
def V_analytical(x):
    return 1220*(1-np.exp(-0.2*x))/(3*x) - (90*(1-np.exp(-0.1*x))/(x))**2


def u(x,a, I):
    return I*np.exp(-a*x)
 
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)
errorCP = []
varCP = []
a_m = cp.E(a)
I_m = cp.E(I)
Nt = 10**2
x = np.linspace(0.000000001, 10, Nt)
dt = x[1] - x[0]


k = 3
m = 2


P = cp.orth_ttr(m, dist)
#for i in range(0,6):
nodes, weights = cp.generate_quadrature(2, dist, rule="C",nested=True,growth=True)
#    print len(nodes[0])

#i1,i2 = np.mgrid[:len(weights), :Nt]
#solves = u(x[i2],nodes[0][i1],nodes[1][i1])

#U_hat = cp.fit_quadrature(P, nodes, weights, solves)

#errorCP.append(dt*np.sum(np.abs(E_analytical(x)) - cp.E(U_hat,dist)))
#varCP.append(dt*np.sum(np.abs(V_analytical(x)) - cp.Var(U_hat,dist)))



#pl.scatter(nodes[0],nodes[1],s=200*weights,alpha=0.5)
pl.scatter(nodes[0],nodes[1])
"""
fig = pl.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(8,8) 
ax0 = pl.subplot(gs[:-1,1:])
ax0.scatter(nodes[0],nodes[1],s=1000*weights,alpha=0.5)

ax1 = pl.subplot(gs[:-1,0])
ax1.scatter(nodes[0,:k+1],nodes[1,:k+1])
ax2 = pl.subplot(gs[-1,1:])
ax2.scatter(nodes[0,::k+1],nodes[1,::k+1])
ax1.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)

pl.tight_layout()
"""

#pl.plot(K,errorCP,linewidth=2)
#pl.plot(K, varCP,linewidth=2)
pl.xlabel("a")
pl.ylabel("I")
#pl.yscale('log')
#pl.title("Error in expectation value and variance")
#pl.legend([Expectation value PC","Variance PC"])#
#pl.savefig("nodes_sparse.png")
pl.show()
