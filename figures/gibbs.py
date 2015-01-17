import chaospy as cp
import pylab as plt
import numpy as np

#I = cp.Uniform()
a = cp.Normal()
#a = cp.Uniform()
#dist = cp.J(a,I)

def U(x):
    return np.piecewise(x, [x < 0, x >= 0], [-1,1])
    
legend=[]
x = plt.linspace(-1,1, 100)

for m in range(5,30,5):
    P = cp.orth_ttr(m, a)
    nodes, weights = cp.generate_quadrature(m, a, rule="G")
    solves = U(nodes[0])
    U_hat = cp.fit_quadrature(P, nodes, weights, solves)

    
    legend.append("M = %d" % (m))
    plt.plot(x,U_hat(x), linewidth=2)

plt.rc("figure", figsize=[6,4])

plt.plot(x,U(x), linewidth=2)
plt.ylim(-1.5,1.5)
plt.xlabel("Y")
plt.ylabel("X")
#plt.title("Orthogonal series expansion of the sign function")
legend.append("True function")
plt.legend(legend, loc=2)
plt.savefig("gibbs.png")

plt.show()
