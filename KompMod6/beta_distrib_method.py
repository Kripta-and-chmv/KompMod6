import random
import numpy as np

def first(u, v):
    """for u, v - integer
    """
    rand_arr = []
    for i in range(u + v - 1):
        rand_arr.append(random.random())

    random_arr.sort()

    return rand_arr[u-1]

def second(u, v):
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