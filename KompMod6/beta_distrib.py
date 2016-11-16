import random
import numpy as np
import scipy as sc

def first_method(u, v):
    """for u, v - integer
    """
    rand_arr = []
    for i in range(u + v - 1):
        rand_arr.append(random.random())

    random_arr.sort()

    return rand_arr[u-1]

def second_method(u, v):
    """for u, v - positive
    """
    q1 = random.random()
    q2 = random.random()
    q1_pow = q1**(1/u)
    q2_pow = q2**(1/v)

    while q1_pow + q2_pow > 1:
        q1 = random.random()
        q2 = random.random()
        q1_pow = q1**(1/u)
        q2_pow = q2**(1/v)

    return q1_pow / (q1_pow + q2_pow)

def bessel(u, v):
    def integ(x, u, v):
        return x**(u - 1) * (1 - x)**(v - 1)
    return sc.integrate.quad(integ, 0, 1, args=(u, v))[0]

def calc_pdf(x, u, v):
    numer1 = (x**(u-1)) 
    numer2 = ((1 - x)**(v-1))
    numer = numer1 * numer2
    denumer = bessel(u, v)
    k = numer / denumer
    return k

def calc_cdf(x, u, v):
    def i_func(x, a, b):
        def integ(x, a, b):
            return x**(a-1) * (1 - x)**(b-1)
        def inc_bet(x, a, b):
            k = sc.integrate.quad(integ, 0, x, args=(a, b))
            return k[0]

        return inc_bet(x, a, b)/sc.special.beta(a, b)
    
    cdf1 = i_func(x, u, v)
    #cdf2 = sc.integrate.quad(calc_pdf, 0, x, args = (u, v))
    return cdf1