from jit_test import ident, ident_parallel
import numpy as np
import time
import matplotlib.pyplot as plt

num = np.linspace(10, 2000, 50)
par = []
nopar = []

nf = 100
aa = ident_parallel(np.arange(nf * nf).reshape((nf, nf)))

for n in num:
    nf = int(np.floor(n))
    arr = np.arange(nf * nf).reshape((nf, nf))

    t = time.time()
    bb = ident(arr)
    nopar.append(time.time() - t)

    t = time.time()
    aa = ident_parallel(arr)
    par.append(time.time() - t)


plt.plot(num, nopar)
plt.plot(num, par)

plt.show()
