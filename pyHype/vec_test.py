import numpy as np
import time

n = 1000
s = 1000
vec1 = np.random.random((4, s))
vec = np.random.random((1, 4*s))


def f(vec):
    return vec * vec + np.sin(vec) / (1 + np.cos(vec))


t = time.time()

for i in range(n):
    a = list(map(f, [row for row in vec1]))

t1 = time.time()

time_small = t1 - t

t = time.time()

for i in range(n):
    e = vec * vec + np.sin(vec) / (1 + np.cos(vec))

t1 = time.time()

time_big = t1 - t

print(time_small, time_big)