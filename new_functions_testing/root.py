import scipy.optimize as spo
import numpy as np

def F(x, theta=15, M=2.0, g=1.4):
    _theta = theta * np.pi / 180
    return np.tan(_theta) * ((M**2) * (g + np.cos(2 * x)) + 2) - 2 * ((M**2) * np.sin(x)**2 - 1) / np.tan(x)


x = spo.broyden1(F, 30 * np.pi / 180)
print(x * 180 / np.pi)
