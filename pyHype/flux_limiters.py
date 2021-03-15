import numpy as np

def van_albada(x):
    return (np.square(x) + x) / (np.square(x) + 1)

def van_leer(x):
    return (np.absolute(x) + x) / (x + 1)
