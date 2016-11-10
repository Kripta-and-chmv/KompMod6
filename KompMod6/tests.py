import scipy as sc
import scipy.special as scsp
import sys
import numpy as np
import fisher
import matplotlib.pyplot as plt

def kramer_smirnov(seq, alpha):
    def calc_s_star(seq):
        len_seq = len(seq)
        s_star = sum([(fisher.calc_cdf(x) - (2 * i - 1) / (2 * len_seq))**2 \
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
    """Тест Хи-квадрат"""
    print("Тест хи квадрат:")

    mod = max(sequence)
    len_seq = len(sequence)
    # разбиваем отрезок от 0 до mod на интервалы
    intervals_amount = int(5 * sc.log10(len_seq))
    K = intervals_amount
    lngth = mod/K   
    intervals = [x * lngth for x in range(0, K+1)]
    
    #определяем количество попаданий в интервалы
    hits_amount = []    
    for a, b in zip(intervals[:-1], intervals[1:]):
            count = sum([a <= x < b for x in sequence])
            hits_amount.append(count)

    emper_prob = [x / len_seq for x in hits_amount]

    # Вычисляется вероятность попадания слчайной величины в заданные
    # интервалы
    def calc_probs(intervals):
        return [fisher.calc_cdf(v, u, x) - fisher.calc_cdf(v, u, y) for x, y in zip(intervals[1:], intervals[:-1])]

    probabils = calc_probs(intervals)

    def graph(intervals, probabils, emper_prob):
        width = intervals[len(intervals) - 1] / (len(intervals) - 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(intervals[:len(intervals) - 1], probabils, width, color="white", label = u'Theoretical')
        ax.bar(intervals[:len(intervals) - 1], emper_prob, width, alpha=0.5, color="black", label = u'Emperical')
        ax.legend(loc = 'best', frameon = True)
        plt.title('Chi2 Histogram')
        plt.xlabel('intervals')
        plt.ylabel('hits amount')
        plt.xticks(intervals)
        plt.show()



    graph(intervals, probabils, emper_prob)
    # вычисляется статистика
    addition = 0
    for hits, probs in zip(hits_amount, probabils):
        if probs == 0: continue
        addition += (hits / len_seq - probs)**2 / probs

    s_star = len(sequence) * addition
    print("\tЗначение статистики - {}".format(s_star))

    # вычисляется P(S*>S)
    r = intervals_amount - 1
    print("\tКоличество степеней свободы - {}".format(r))

    def integrand(x, r):
        return x ** (r / 2 - 1) * sc.exp(-x / 2)

    prob_s = sc.integrate.quad(integrand, s_star, np.inf, args = (r))
    prob_s = prob_s[0] / (2 ** (r / 2) * scsp.gamma(r / 2))

    print("\tP(S*>S) - {}".format(prob_s))
    print("\tПрохождение теста хи квадрат - {}\n".format(prob_s > alpha))