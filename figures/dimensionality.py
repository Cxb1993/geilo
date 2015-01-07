import chaospy as cp
import pylab as pl

m = 5
d = 10

M = pl.arange(1, m+1)
D = pl.arange(1, d+1)
legend = []

#pl.xkcd()
pl.figure()
for m in M:
    N = [cp.terms(m, d) for d in D]
    pl.plot(D,N,linewidth=2)
    legend.append("M = %d" % (m))
    
pl.xlabel("Dimensions, D")
pl.ylabel("Terms, N")
pl.title("Number of terms for a polynomial")
pl.legend(legend, loc=2)
pl.axis([1,d,1,500])
pl.savefig("dimensionality.png")

pl.show()
