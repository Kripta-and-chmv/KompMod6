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
    length = 200

    seq_bet = [beta.second_method(u, v) for i in range(length)]

    tests.chisqr_test_bet(seq_bet, alpha, u, v)
    tests.kolmagorov_bd(seq_bet, alpha, u, v)

    seq_fish = [fisher.second_method(u, v) for i in range(length)]
    tests.chisqr_test_fish(seq_fish, alpha, u, v)
    tests.kolmagorov_fisher(seq_fish, alpha, u, v)

if __name__ == "__main__":
    main()