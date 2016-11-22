import numpy as np
import time
import tests
import scipy as sc
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import beta_distrib as beta
import fisher

def get_arguments():
    with open("arguments.txt", "r") as f:
        file_str = f.read()
        args = file_str.split(" ")
        u, v, alpha = float(args[0]), float(args[1]), float(args[2])
        return u, v, alpha



def main():
    sys.stdout = open("output.txt", "w+")

    u, v, alpha = get_arguments()
    length = 1000

    seq_bet = [beta.second_method(u, v) for i in range(length)]

    tests.chisqr_test_bet(seq_bet, alpha, u, v)
    #tests.smirnov_beta(seq_bet, alpha, u, v)

    seq_fish = [fisher.second_method(u, v) for i in range(length)]
    tests.chisqr_test_fish(seq_fish, alpha, u, v)
    tests.kolmagorov(seq_fish, alpha, u, v)

if __name__ == "__main__":
    main()