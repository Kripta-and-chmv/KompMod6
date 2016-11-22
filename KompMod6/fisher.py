import scipy as sc
import numpy as np
import scipy.special as scsp
import beta_distrib as bd


def bessel(u, v):
    def integ(x, u, v):
        return x**(u - 1) * (1 - x)**(v - 1)
    return sc.integrate.quad(integ, 0, 1, args=(u, v))[0]

def calc_pdf(x, u, v):
    numer = x**(u/2 - 1) * u**(u / 2) * v**(v / 2)
    denumer = scsp.beta(u/2, v/2) * (v + u*x)**((u + v) / 2)
    return numer / denumer

def calc_cdf(x, u, v):

    def i_func(x, a, b):
        def integ(x, a, b):
            return x**(a-1) * (1 - x)**(b-1)
        def inc_bet(x, a, b):
            k = sc.integrate.quad(integ, 0, x, args=(a, b))
            return k[0]

        return inc_bet(x, a, b)/scsp.beta(a, b)

    #cdf1 = sc.integrate.quad(calc_pdf, 0, x, args = (u, v))
    cdf2 = i_func(u*x/(u*x + v), u/2, v/2)

    return cdf2

def second_method(u, v):
    y = bd.second_method(u/2, v/2)
    return y / (1 - y)