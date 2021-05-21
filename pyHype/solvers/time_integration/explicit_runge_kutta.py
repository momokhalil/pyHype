import numpy as np
from pyHype.solvers.time_integration.base import ExplicitRungeKutta

class RK4(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the fourth order explicit Runge-Kutta method for time integrations.

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """

        super().__init__(inputs, refBLK)

        self.a = [[0.5], [0, 0.5], [0, 0, 1], [1/6, 1/3, 1/3, 1/6]]
        self.num_stages = 4

class RK2(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the fourth order explicit Runge-Kutta method for time integrations.

        Butcher Tableau:
        1/2
        0   1

        """

        super().__init__(inputs, refBLK)

        self.a = [[0.5], [0, 1]]
        self.num_stages = 2
