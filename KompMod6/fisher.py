import scipy as sc
import numpy as np
import scipy.special as scsp

def calc_pdf(x, u, v):
    numer = scsp.gamma((u + v) / 2) * u**(u / 2) * v**(v / 2) *\
       x**(u / 2 - 1)
    denumer = scsp.gamma(u / 2) * scsp.gamma(v / 2) *\
        (v + u * x)**((u + v) / 2) * scsp.beta(u/2, v/2)
    return numer / denumer

def calc_cdf(x, u, v):

    def i_func(x, a, b):
        def integ(x, a, b):
            return x**(a-1) * (1 - x)**(b-1)
        def inc_bet(x, a, b):
            k = sc.integrate.quad(integ, 0, x, args=(a, b))
            return k[0]

        return inc_bet(x, a, b)/scsp.beta(a, b)

    cdf1 = sc.integrate.quad(calc_pdf, -np.inf, x, args = (u, v))
    cdf2 = i_func(u*x/(u*x + v), u/2, v/2)

    return cdf1[0]