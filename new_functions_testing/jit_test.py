import numpy as np
import numba as nb
from time import time
from profilehooks import profile

@nb.njit(parallel=True, cache=True)
def ident_parallel(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2

def ident(x):
    return np.cos(x) ** 2 + np.sin(x) ** 2
