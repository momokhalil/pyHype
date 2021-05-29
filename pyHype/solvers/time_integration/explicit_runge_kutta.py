"""
Copyright 2021 Mohamed Khalil

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from pyHype.solvers.time_integration.base import TimeIntegrator


class ExplicitRungeKutta(TimeIntegrator):
    def __init__(self,
                 inputs,
                 refBLK,
                 a: list[list] = None,
                 num_stages: int = None,
                 ):

        # Call superclass constructor
        super().__init__(inputs, refBLK)
        # Butcher tableau representation
        self.a = a
        # Number of stages
        self.num_stages = num_stages

    def integrate(self, dt):

        # Save state vector
        U = self.refBLK.state.U.copy()
        # Initialise dictionary to store stage residuals
        _stage_residuals = {}
        # Iterate across num_stages
        for stage in range(self.num_stages):
            # Get residual for current stage
            _stage_residuals[stage] = self.get_residual()
            # Copy U into intermediate stage
            _intermediate_state = U
            # Add stage contributions to current stage according to the method's Butcher tableau
            for step in range(stage + 1):
                # Only add if tableau factor is non-zero
                if self.a[stage][step] != 0:
                    # Accumulate stage contributions
                    _intermediate_state = _intermediate_state + dt * self.a[stage][step] * _stage_residuals[step]
            # Update state using intermediate state
            self.update_state(_intermediate_state)

    @classmethod
    def ExplicitEuler1(cls, inputs, refBLK):
        """
        Defines the Explicit-Euler explicit Runge-Kutta method for time integration.
        Order: 1

        Butcher Tableau:
        1

        """
        return cls(inputs,
                   refBLK,
                   a=[[1]],
                   num_stages=1)

    @classmethod
    def Generic2(cls, inputs, refBLK):
        """
        Defines the generic second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        a
        1-1/(2a)    1/(2*a)

        """
        a = inputs.alpha
        return cls(inputs,
                   refBLK,
                   a=[[a],
                      [1 - 1 / (2 * a), 1 / (2 * a)]],
                   num_stages=2)

    @classmethod
    def RK2(cls, inputs, refBLK):
        """
        Defines the classical second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        1/2
        0   1

        """
        return cls(inputs,
                   refBLK,
                   a=[[0.5],
                      [0, 1]],
                   num_stages=2)


    @classmethod
    def Ralston2(cls, inputs, refBLK):
        """
        Defines the Ralston explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        2/3
        1/4 3/4

        """
        return cls(inputs,
                   refBLK,
                   a=[[2 / 3],
                      [1 / 4, 3 / 4]],
                   num_stages=2)


    @classmethod
    def Generic3(cls, inputs, refBLK):
        """
        Defines the generic third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        a
        1+k             -k
        0.5 - 1/(6*a)   1/(6a(1-a))   (2-3a)/(6(1-a))]]

        """
        a = inputs.alpha

        if a == 0 or a == 2 / 3 or a == 1:
            raise ValueError('Value of alpha parameter is not allowd.')
        else:
            k = (1 - a) / a / (3 * a - 2)
            return cls(inputs,
                       refBLK,
                       a=[[a],
                          [1 + k, -k],
                          [0.5 - 1 / (6 * a), 1 / (6 * a * (1 - a)), (2 - 3 * a) / (6 * (1 - a))]],
                       num_stages=3)


    @classmethod
    def RK3(cls, inputs, refBLK):
        """
        Defines the classical third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        -1	2
        1/6	2/3	1/6

        """
        return cls(inputs,
                   refBLK,
                   a=[[0.5],
                      [-1, 2],
                      [1 / 6, 2 / 3, 1 / 6]],
                   num_stages=3)


    @classmethod
    def RK3SSP(cls, inputs, refBLK):
        """
        Defines the Strong Stability Preserving (SSP) third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1
        1/4 1/4
        1/6	1/6	2/3

        """
        return cls(inputs,
                   refBLK,
                   a=[[1],
                      [1 / 4, 1 / 4],
                      [1 / 6, 1 / 6, 2 / 3]],
                   num_stages=3)


    @classmethod
    def Ralston3(cls, inputs, refBLK):
        """
        Defines the Ralston third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        0   3/4
        2/9 1/3 4/9

        """
        return cls(inputs,
                   refBLK,
                   a=[[1 / 2],
                      [0, 3 / 4],
                      [2 / 9, 1 / 3, 4 / 9]],
                   num_stages=3)

    @classmethod
    def RK4(cls, inputs, refBLK):
        """
        Defines the fourth order explicit Runge-Kutta method for time integrations.

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """
        return cls(inputs,
                   refBLK,
                   a=[[0.5],
                      [0, 0.5],
                      [0, 0, 1],
                      [1 / 6, 1 / 3, 1 / 3, 1 / 6]],
                   num_stages=4)


    @classmethod
    def Ralston4(cls, inputs, refBLK):
        """
        Defines the Ralston fourth order explicit Runge-Kutta method for time integrations.
        Order: 4

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """
        return cls(inputs,
                   refBLK,
                   a=[[0.4],
                      [0.29697761, 0.15875964],
                      [0.21810040, -3.05096516, 3.83286476],
                      [0.17476028, -.55148066, 1.20553560, 0.17118478]],
                   num_stages=4)


    @classmethod
    def DormandPrince5(cls, inputs, refBLK):
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
        return cls(inputs,
                   refBLK,
                   a=[[1 / 5],
                      [3 / 40, 9 / 40],
                      [44 / 45, -56 / 15, 32 / 9],
                      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
                      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
                      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]],
                   num_stages=6)
