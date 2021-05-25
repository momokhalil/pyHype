import numpy as np
from numba import njit

@njit()
def van_albada(x):
    return (np.square(x) + x) / (np.square(x) + 1)

@njit()
def van_leer(x):
    return (np.absolute(x) + x) / (x + 1)
