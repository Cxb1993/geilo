import chaospy as cp
import pylab as pl

d = 4
n = 10

N = pl.arange(1, n+1)
D = pl.arange(1, d+1)
legend = []

def k(n, D):
    return (n+1)**D

#pl.xkcd()
pl.figure()
for d in D:
    pl.plot(N, k(N,d),linewidth=2)
    legend.append("D = %d" % (d))
    
pl.xlabel("Quadrature order, L")
pl.ylabel("Total order, K")
#pl.title("Number of terms for a polynomial")
pl.rc("figure", figsize=[6,4])

pl.legend(legend, loc=2)
pl.axis([1,n,1,500])
pl.savefig("dimensionality_nodes.png")

pl.show()
