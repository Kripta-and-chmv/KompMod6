import numpy as np
import time
import tests
import scipy as sc
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import beta_distrib_method as method

def get_arguments():
    with open("arguments.txt", "r") as f:
        file_str = f.read()
        args = file_str.split(" ")
        u, v, alpha = float(args[0]), float(args[1]), float(args[2])
        return u, v, alpha



def main():
    sys.stdout = open("output.txt", "w+")

    #u, v, alpha = get_arguments()
    
    method.first(1, 1)


if __name__ == "__main__":
    main()