import chaospy as cp
import pylab as pl
import numpy as np


m = 0

l = 4

L = pl.arange(m+1, l+1)
D =pl.array([2,3,4])
legend = []

style = ["ro","bo","go","yo"]


L = pl.arange(m+1, l+1)
L_i = L**2+1
L_i[0] = 1
pl.figure()
i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        print l
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="E",sparse=True,growth=False)
        k.append(len(nodes[0]))
        
    pl.plot(L, k,style[i]+"-", linewidth=2)
    legend.append("D = %d" % (d))
    i+=1



i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="E",sparse=True,growth=True)
        k.append(len(nodes[0]))
        
    pl.plot(L_i, k,style[i]+"-.", linewidth=2)
    legend.append("D = %d, G" % (d))
    i+=1


i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="C",sparse=True, growth=True)
        k.append(len(nodes[0]))
        
    pl.plot(L_i, k,style[i] + "--",linewidth=2)
    legend.append("D = %d, C" % (d))
    i+=1
    
pl.xlabel("Terms, N")
pl.ylabel("Nodes, k")
pl.yscale('log')

pl.legend(legend, loc=2)
#pl.axis([m,l,1,2000])
pl.savefig("dimensionality_nodes_nested.png")

pl.show()
