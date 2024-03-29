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
from __future__ import annotations

import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import numba as nb

from pyhype.utils.logger import Logger
from pyhype.time_marching.base import TimeIntegrator

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.blocks.base import Blocks


class ExplicitRungeKutta(TimeIntegrator):
    def __init__(
        self,
        config,
        a: list[list] = None,
    ):

        super().__init__(config)
        self.a = a
        self.num_stages = len(a)

        self._logger = Logger(config=config)

    def integrate(self, dt: float, blocks: Blocks) -> None:
        """
        Perform integration for a general explicit Runge-Kutta scheme based on the Butcher tableau.

        Parameters:
            - parent_block (QuadBlock): Reference block whos state is to be integrated in time
            - dt (float): Time step

        Return:
            - N/A
        """
        stage_residuals = {i: {} for i in blocks.blocks.keys()}
        U = {i: block.state.data.copy() for i, block in blocks.blocks.items()}
        temp_state = {
            i: np.zeros_like(block.state.data) for i, block in blocks.blocks.items()
        }
        for stage in range(self.num_stages):
            for i, block in reversed(blocks.blocks.items()):
                stage_residuals[i][stage] = block.dUdt()
                intermediate_state = U[i]
                for step in range(stage + 1):
                    if self.a[stage][step] != 0:
                        intermediate_state = self._update_state(
                            temp_state[i],
                            intermediate_state,
                            dt * self.a[stage][step],
                            stage_residuals[i][step],
                        )
                block.state.data = intermediate_state

            blocks.apply_boundary_condition()
            for i, block in blocks.blocks.items():
                block.clear_cache()
                block.recon_block.clear_cache()

    @staticmethod
    @nb.njit(cache=True)
    def _update_state(update, state, dta, residual):
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                for k in range(state.shape[2]):
                    update[i, j, k] = state[i, j, k] + dta * residual[i, j, k]
        return update

    @classmethod
    def ExplicitEuler1(cls, config):
        """
        Defines the Explicit-Euler explicit Runge-Kutta method for time integration.
        Order: 1

        Butcher Tableau:
        1

        """
        return cls(config, a=[[1]])

    @classmethod
    def Generic2(cls, config):
        """
        Defines the generic second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        a
        1-1/(2a)    1/(2*a)

        """
        a = config.alpha
        return cls(config, a=[[a], [1 - 1 / (2 * a), 1 / (2 * a)]])

    @classmethod
    def RK2(cls, config):
        """
        Defines the classical second order explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        1/2
        0   1

        """
        return cls(config, a=[[0.5], [0, 1]])

    @classmethod
    def Ralston2(cls, config):
        """
        Defines the Ralston explicit Runge-Kutta method for time integration.
        Order: 2

        Butcher Tableau:
        2/3
        1/4 3/4

        """
        return cls(config, a=[[2 / 3], [1 / 4, 3 / 4]])

    @classmethod
    def Generic3(cls, config):
        """
        Defines the generic third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        a
        1+k             -k
        0.5 - 1/(6*a)   1/(6a(1-a))   (2-3a)/(6(1-a))]]

        """
        a = config.alpha

        if a in (0, 2 / 3, 1):
            raise ValueError("Value of alpha parameter is not allowd.")

        k = (1 - a) / a / (3 * a - 2)
        return cls(
            config,
            a=[
                [a],
                [1 + k, -k],
                [0.5 - 1 / (6 * a), 1 / (6 * a * (1 - a)), (2 - 3 * a) / (6 * (1 - a))],
            ],
        )

    @classmethod
    def RK3(cls, config):
        """
        Defines the classical third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        -1	2
        1/6	2/3	1/6

        """
        return cls(config, a=[[0.5], [-1, 2], [1 / 6, 2 / 3, 1 / 6]])

    @classmethod
    def RK3SSP(cls, config):
        """
        Defines the Strong Stability Preserving (SSP) third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1
        1/4 1/4
        1/6	1/6	2/3

        """
        return cls(config, a=[[1], [1 / 4, 1 / 4], [1 / 6, 1 / 6, 2 / 3]])

    @classmethod
    def Ralston3(cls, config):
        """
        Defines the Ralston third order explicit Runge-Kutta method for time integration.
        Order: 3

        Butcher Tableau:
        1/2
        0   3/4
        2/9 1/3 4/9

        """
        return cls(config, a=[[1 / 2], [0, 3 / 4], [2 / 9, 1 / 3, 4 / 9]])

    @classmethod
    def RK4(cls, config):
        """
        Defines the fourth order explicit Runge-Kutta method for time integrations.

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """
        return cls(config, a=[[0.5], [0, 0.5], [0, 0, 1], [1 / 6, 1 / 3, 1 / 3, 1 / 6]])

    @classmethod
    def Ralston4(cls, config):
        """
        Defines the Ralston fourth order explicit Runge-Kutta method for time integrations.
        Order: 4

        Butcher Tableau:
        1/2
        0	1/2
        0	0	1
        1/6	1/3	1/3	1/6

        """
        return cls(
            config,
            a=[
                [0.4],
                [0.29697761, 0.15875964],
                [0.21810040, -3.05096516, 3.83286476],
                [0.17476028, -0.55148066, 1.20553560, 0.17118478],
            ],
        )

    @classmethod
    def DormandPrince5(cls, config):
        """
        Defines the Dormand-Prince fifth order explicit Runge-Kutta method for time integrations.
        Order: 5

        Butcher Tableau:
        1/5
        3/40	    9/40
        44/45	    −56/15	    32/9
        19372/6561	−25360/2187	64448/6561	−212/729
        9017/3168	−355/33	    46732/5247	49/176	    −5103/18656
        35/384	0	500/1113	125/192	    −2187/6784	11/84

        """
        return cls(
            config,
            a=[
                [1 / 5],
                [3 / 40, 9 / 40],
                [44 / 45, -56 / 15, 32 / 9],
                [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
                [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
                [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
            ],
        )
