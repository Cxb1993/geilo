import chaospy as cp
import pylab as plt
import numpy as np

def u(x,a, I):
    return I*np.exp(-a*x)

a = cp.Uniform(0, 2)
I = cp.Uniform(0, 2)
dist = cp.J(a,I)

M = 10

P = cp.orth_ttr(M, dist)
N = len(P)
q = cp.variable(2)[0]

P_nk = cp.outer(P, P)
result = cp.E(q*P_nk,dist)

result[result>10**-8] = 1
result[result<=10**-8] = 0
plt.matshow(result, cmap=plt.cm.gray)
plt.savefig("binary_matrix.png")
plt.show()
