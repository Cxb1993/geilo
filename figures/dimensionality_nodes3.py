import chaospy as cp
import pylab as pl
import numpy as np


m = 1

legend = []

style = ["r","b","g","y"]
pl.figure()

pl.plot(-1,1, "k-")
pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "b")
pl.plot(-1,1, "g")
pl.plot(-1,1, "y")
pl.legend(["Legendre","Clenshaw-Curtis", "D=2","D=3","D=5","D=7"],loc=2)


D = [2,3,5,7]
l = 6
L = pl.arange(m+1, l+1)
L_i = L**2+1
L_i[0] = 1


i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="E",sparse=True,growth=False)
        k.append(len(nodes[0]))
        
    pl.plot(L, k,style[i]+"-", linewidth=2)
#    legend.append("D = %d, G" % (d))
    i+=1


i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="C",sparse=True, growth=False)
        k.append(len(nodes[0]))
        
    pl.plot(L, k,style[i] + "--",linewidth=2)
#    legend.append("D = %d, C" % (d))
    i+=1

pl.rc("figure", figsize=[6,4])
    
pl.xlabel("Terms, N")
pl.ylabel("Nodes, k")
pl.yscale('log')
pl.xlim([2,6])
pl.ylim([10**0,10**6])
#pl.legend(legend, loc=2)
#pl.axis([m,l,1,2000])
pl.savefig("dimensionality_nodes_sparse.png")

pl.show()
