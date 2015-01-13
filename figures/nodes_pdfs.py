import chaospy as cp
import pylab as pl
import numpy as np
from matplotlib import gridspec


k = 4
m = 2
    
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)

P = cp.orth_ttr(m, dist)
nodes, weights = cp.generate_quadrature(k, dist, rule="G")

pl.figure()
pl.scatter(nodes[0],nodes[1],s=1000*weights,alpha=0.5)
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("nodes_uniform.png")



a = cp.Normal(0, 0.1)
I = cp.Normal()
dist = cp.J(a,I)

P = cp.orth_ttr(m, dist)
nodes, weights = cp.generate_quadrature(k, dist, rule="G")

pl.figure()
pl.scatter(nodes[0],nodes[1],s=2000*weights,alpha=0.5)
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("nodes_normal.png")
pl.show()

a = cp.Gamma(2,1)
I = cp.Gamma()
dist = cp.J(a,I)

P = cp.orth_ttr(m, dist)
nodes, weights = cp.generate_quadrature(k, dist, rule="G")

pl.figure()
pl.scatter(nodes[0],nodes[1],s=2000*weights,alpha=0.5)
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("nodes_gamma.png")
pl.show()

a = cp.Beta(2,2,2,3)
I = cp.Beta(1,3)
dist = cp.J(a,I)

P = cp.orth_ttr(m, dist)
nodes, weights = cp.generate_quadrature(k, dist, rule="G")

pl.figure()
pl.scatter(nodes[0],nodes[1],s=2000*weights,alpha=0.5)
pl.xlabel("a")
pl.ylabel("I")
pl.savefig("nodes_beta.png")
pl.show()


"""
fig = pl.figure(figsize=(8, 6)) 
gs = gridspec.GridSpec(8,8) 
ax0 = pl.subplot(gs[:-1,1:])
ax0.scatter(nodes[0],nodes[1],s=2000*weights,alpha=0.5)

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
