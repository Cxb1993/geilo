import chaospy as cp
import pylab as pl
import numpy as np


m = 0

l = 8

L = pl.arange(m+1, l+1)
D =pl.array([2,3,4,6])

style = ["r","b","g","y"]
pl.figure()

legend = []
pl.plot(-1,1, "k-")
pl.plot(-1,1, "k--")
pl.plot(-1,1, "r")
pl.plot(-1,1, "b")
pl.plot(-1,1, "g")
pl.plot(-1,1, "y")
pl.legend(["Full tensor grid","Sparse grid", "D=2","D=3","D=4","D=6"],loc=2,prop={"size" :12})
pl.rc("figure", figsize=[6,4])


L = pl.arange(m+1, l+1)
L_i = L**2+1
L_i[0] = 1
i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        print l
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="G",sparse=False,growth=False)
        k.append(len(nodes[0]))
        
    pl.plot(L, k,style[i]+"-", linewidth=2)
#    legend.append("D = %d" % (d))
    i+=1

i=0
for d in D:
    dist = cp.Iid(cp.Uniform(), d)
    k = []
    for l in L:
        print l
        P = cp.orth_ttr(m, dist)
        nodes, weights = cp.generate_quadrature(l, dist, rule="G",sparse=True,growth=False)
        k.append(len(nodes[0]))
        
    pl.plot(L, k,style[i]+"--", linewidth=2)
#    legend.append("D = %d" % (d))
    i+=1

pl.rc("figure", figsize=[6,4])
    

pl.xlabel("Quadrature order, L")
pl.ylabel("Number of nodes, K")
pl.yscale('log')
pl.xlim([1,8])
pl.ylim([10**0,10**6])
#pl.legend(legend, loc=2)
#pl.axis([m,l,1,2000])
pl.savefig("dimensionality_nodes_gq_sparse.png")

pl.show()
