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
    
pl.xlabel("Terms, N")
pl.ylabel("Nodes, k")
#pl.title("Number of terms for a polynomial")

pl.legend(legend, loc=2)
pl.axis([1,n,1,500])
pl.savefig("dimensionality_nodes.png")

pl.show()
