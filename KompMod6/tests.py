import scipy as sc
import scipy.special as scsp
import sys
import numpy as np
import fisher

def kramer_smirnov(seq, alpha):
    def calc_s_star(seq):
        len_seq = len(seq)
        s_star = sum([(fisher.pdf(x) - (2 * i - 1) / (2 * len_seq))**2 \
            for x, i in zip(seq, range(1, len_seq+1))])
        return s_star / (12 * len_seq)
    
    def i_func(v, z):
        k = 0
        return sc.sum((z / 2)**(v + 2*k) / (scsp.gamma(k + 1) * scsp.gamma(k + v + 1)), k)

    def calc_a1(s_star):
        j = 0
        t = (4*j + 1)**2 / (16 * s_star)
        inf_sum = ((scsp.gamma(j + 0.5) * np.sqrt(4*j + 1) *\
            np.exp(-(4*j + 1)**2 / (16 * s_star))) *\
           (i_func(-1/4, (4*j + 1)**2 / (16 * s_star)) -\
           i_func(1/4, (4*j + 1)**2 / (16 * s_star))) , j)

        return inf_sum / np.sqrt(2 * s_star)

    seq.sort()

    s_star = calc_s_star(seq)
    a1 = calc_a1(s_star)

    hit = 1 - a1 > alpha
    
    return hit

def chisqr_test(sequence, alpha, v, u):
    

    mod = max(sequence)
    len_seq = len(sequence)
    
    intervals_amount = int(5 * sc.log10(len_seq))
    K = intervals_amount
    lngth = mod/K   
    intervals = [x * lngth for x in range(0, K+1)]
    
    
    hits_amount = []    
    for a, b in zip(intervals[:-1], intervals[1:]):
            count = sum([a <= x < b for x in sequence])
            hits_amount.append(count)

    
    def calc_probs(intervals):
        return [norm_d.cdf(v, u, x) - norm_d.cdf(v, u, y) for x, y in zip(intervals[1:], intervals[:-1])]

    probabils = calc_probs(intervals)

    
    addition = 0
    for hits, probs in zip(hits_amount, probabils):
        if probs == 0: continue
        addition += (hits / len_seq - probs)**2 / probs

    s_star = len(sequence) * addition
    

    
    r = intervals_amount - 1
    

    def integrand(x, r):
        return x ** (r / 2 - 1) * np.exp(-x / 2)

    prob_s = sc.integrate.quad(integrand, s_star, np.inf, args = (r))
    prob_s = prob_s[0] / (2 ** (r / 2) * sc.special.gamma(r / 2))

    hit = prob_s > alpha
    
    return hit