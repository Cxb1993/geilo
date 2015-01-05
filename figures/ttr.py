import chaospy as cp
import pylab as plt

M = range(1,20)
e = []

a = cp.Normal()

for m in M:
    P, norm = cp.orth_ttr(m, a, retall=True)
    #e.append(cp.E(P**2,a))
    e.append(norm)
    #print cp.E(P**2,a)

print M
print e
plt.plot(e,M)
plt.show()
