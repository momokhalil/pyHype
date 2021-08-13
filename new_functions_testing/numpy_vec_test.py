import numpy as np
import time
import numba as nb

n = 700
itr = 1000

a = np.random.random((n, n))
b = np.random.random((n, n))

def timeit(func):
    def wrapper(a, b):
        _ = func(a, b)
        t = time.time()
        for i in range(itr):
            d = func(a, b)
        t1 = time.time()
        print(t1 - t)
        return d
    return wrapper

@timeit
def mult(a, b):
    return a ** 2 + b ** 2

@timeit
@nb.njit(cache=True)
def mult_loop_jit(a, b):
    d = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            d[i, j] = a[i, j] ** 2 + b[i, j] ** 2
    return d

@timeit
@nb.njit(cache=True)
def mult_loop_jit2(a, b):
    d = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            d[i, j] = a[i, j] * a[i, j] + b[i, j] * b[i, j]
    return d


d = mult(a, b)
d = mult_loop_jit(a, b)
d = mult_loop_jit2(a, b)
