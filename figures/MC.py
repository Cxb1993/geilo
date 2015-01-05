import chaospy as cp
import numpy as np

def u(x, a, I):
    return I*np.exp(-a*x)
N = 1000
a = cp.Uniform()
I = cp.Uniform()
samplesI = I.sample(N)
samplesa = a.sample(N)
x = np.linspace(0, 10, N)

#dist = cp.J(a,I)
#cp.Var(np.exp(x),dist)
#x= cp.variable()

U = u(x, samplesa, samplesI)

E = np.sum(U)/N
Var = np.sum(U**2)/N - E**2

print "E = ", E
print "Var = ", Var
