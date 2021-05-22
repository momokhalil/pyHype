import numpy as np
from pyHype.solvers.time_integration.base import ExplicitRungeKutta


class ExplicitEuler1(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Explicit-Euler explicit Runge-Kutta method for time integration.
        Order: 1

        Butcher Tableau:
        1

        """

        super().__init__(inputs, refBLK)

        self.a = [[1]]
        self.num_stages = 1


class Generic2(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the generic second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        a
        1-1/(2a)    1/(2*a)

        """

        super().__init__(inputs, refBLK)

        a = inputs.alpha

        self.a = [[a],
                  [1 - 1/(2*a), 1/(2*a)]]
        self.num_stages = 2


class RK2(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the classical second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        1/2
        0   1

        """

        super().__init__(inputs, refBLK)

        self.a = [[0.5],
                  [0, 1]]
        self.num_stages = 2


class Ralston2(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Ralston explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        2/3
        1/4 3/4

        """

        super().__init__(inputs, refBLK)

        self.a = [[2/3],
                  [1/4, 3/4]]
        self.num_stages = 2


class Generic3(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the generic third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        a
        1+k             -k
        0.5 - 1/(6*a)   1/(6a(1-a))   (2-3a)/(6(1-a))]]

        """

        super().__init__(inputs, refBLK)

        a = inputs.alpha

        if a == 0 or a == 2/3 or a == 1:
            raise ValueError('Value of alpha parameter is not allowd.')
        else:
            k = (1 - a)/a/(3*a - 2)
            self.a = [[a],
                      [1 + k, -k],
                      [0.5 - 1/(6*a), 1/(6*a*(1-a)), (2-3*a)/(6*(1-a))]]
            self.num_stages = 3


class RK3(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the classical third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        -1	2
        1/6	2/3	1/6

        """

        super().__init__(inputs, refBLK)

        self.a = [[0.5],
                  [-1, 2],
                  [1/6, 2/3, 1/6]]

        self.num_stages = 3


class RK3SSP(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Strong Stability Preserving (SSP) third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1
        1/4 1/4
        1/6	1/6	2/3

        """

        super().__init__(inputs, refBLK)

        self.a = [[1],
                  [1/4, 1/4],
                  [1/6, 1/6, 2/3]]

        self.num_stages = 3


class Ralston3(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Ralston third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        0   3/4
        2/9 1/3 4/9

        """

        super().__init__(inputs, refBLK)

        self.a = [[1/2],
                  [0, 3/4],
                  [2/9, 1/3, 4/9]]

        self.num_stages = 3


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

        self.a = [[0.5],
                  [0, 0.5],
                  [0, 0, 1],
                  [1/6, 1/3, 1/3, 1/6]]
        self.num_stages = 4


class Ralston4(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Ralston fourth order explicit Runge-Kutta method for time integrations.
        Order: 4

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """

        super().__init__(inputs, refBLK)

        self.a = [[0.4],
                  [0.29697761, 0.15875964],
                  [0.21810040, -3.05096516, 3.83286476],
                  [0.17476028, -.55148066, 1.20553560, 0.17118478]]
        self.num_stages = 4


class DormandPrince5(ExplicitRungeKutta):
    def __init__(self, inputs, refBLK):
        """
        Defines the Dormand-Prince fifth order explicit Runge-Kutta method for time integrations.
        Order: 4

        Butcher Tableau:
        1/5
        3/40	    9/40
        44/45	    −56/15	    32/9
        19372/6561	−25360/2187	64448/6561	−212/729
        9017/3168	−355/33	    46732/5247	49/176	    −5103/18656
        35/384	0	500/1113	125/192	    −2187/6784	11/84

        """

        super().__init__(inputs, refBLK)

        self.a = [[1/5],
                  [3/40, 9/40],
                  [44/45, -56/15, 32/9],
                  [19372/6561, -25360/2187,	64448/6561,	-212/729],
                  [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
                  [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]
        self.num_stages = 6
