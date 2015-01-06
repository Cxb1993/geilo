
import chaospy as cp
import pylab as pl
import numpy as np

def u(x,a, I):
  return I*np.exp(-a*x)
 
a = cp.Uniform(0, 0.1)
I = cp.Uniform(8, 10)
dist = cp.J(a,I)
x = np.linspace(0, 10, 100)
dt = x[1] - x[0]
m = 2

P = cp.orth_ttr(m, dist)
nodes, weights = cp.generate_quadrature(m+1, dist, rule="G")
i1,i2 = np.mgrid[:len(weights), :100]
solves = u(x[i2],nodes[0][i1],nodes[1][i1])
U_hat = cp.fit_quadrature(P, nodes, weights, solves)
print U_hat

"""
a = cp.Uniform(0,0.1)

def u(x,a):
  ax = np.outer(a,x)
  return np.exp(-ax)

m = 2
x = np.linspace(0, 10, 100)

P, norm = cp.orth_ttr(m, a, retall=True)
nodes, weights = cp.generate_quadrature(m+1, a, rule="G")
solves = u(x,nodes[0])
U_hat, c = cp.fit_quadrature(P, nodes, weights,solves,retall=True)
print c
"""
