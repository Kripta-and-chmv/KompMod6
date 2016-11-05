import scipy as sc
import scipy.special as scsp

def pdf(u, v, x):
    numer = scsp.gamma((u + v) / 2) * u**(u / 2) * v**(v / 2) *\
       x**(u / 2 - 1)
    denumer = scsp.gamma(u / 2) * scsp.gamma(v / 2) *\
        (v + u * x)**((u + v) / 2) * scsp.beta(u/2, v/2)
    return numer / denumer