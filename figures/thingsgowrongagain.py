import chaospy as cp
import numpy as np

M = np.linspace(1,20,21)
e = np.zeros(len(M))
a = cp.Normal()

for m in M:
    P = cp.orth_ttr(int(m), a)
    print P

#cp.E()
