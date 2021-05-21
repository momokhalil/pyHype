import numpy as np
from pyHype.solvers.time_integration.base import ExplicitRungeKutta

class RK4(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):

        super().__init__(inputs, refBLK)

        self.a = [[0.5], [0, 0.5], [0, 0, 1], [1/6, 1/3, 1/3, 1/6]]
        self.num_stages = 4

class RK2(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):

        super().__init__(inputs, refBLK)

        self.a = [[0.5]]
        self.num_stages = 2
