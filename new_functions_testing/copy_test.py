import numpy as np
from copy import copy, deepcopy

class Cont:
    def __init__(self, n):
        self.E = np.zeros(n)
        self.W = 4 * np.ones(2 * n)


class Test:
    def __init__(self, n):
        self.a = np.zeros(n)
        self.b = np.ones(2 * n)
        self.c = Cont(n)


a = Test(4)
b = copy(a)
c = deepcopy(a)

print(a.c is b.c)
print(a.c.E is b.c.E)
print(a.c.E is c.c.E)

c.c.E[0] = 3334
print(a.c.E[0], b.c.E[0], c.c.E[0])

b.c.E[0] = 5635
print(a.c.E[0], b.c.E[0], c.c.E[0])

print('AAAAA')
