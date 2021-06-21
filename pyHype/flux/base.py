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
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from abc import abstractmethod
from pyHype.states.states import PrimitiveState, RoePrimitiveState


class FluxFunction:
    def __init__(self, inputs):
        self.inputs = inputs
        self.g = inputs.gamma
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.n = inputs.n


    @staticmethod
    def wavespeeds_x(W: PrimitiveState) -> [np.ndarray]:
        """
        Calculates the slow and fast wavespeeds u - a and u + a.

        Parameters:
            - W (PrimitiveState): PrimitiveState used to calculate the wavespeeds.

        Returns:
            - slow (np.ndarray): Array that stores the slow wavespeeds
            - fast (np.ndarray): Array that stores the fast wavespeeds
        """

        # Check if PrimitiveState
        if isinstance(W, PrimitiveState):
            # Speed of sound
            a = W.a()
            # Compute wavespeeds
            slow, fast = W.u - a, W.u + a
            return slow, fast
        else:
            raise TypeError('Input is not PrimitiveState.')


    def harten_correction_x(self,
                            Wroe: RoePrimitiveState,
                            WL: PrimitiveState,
                            WR: PrimitiveState
                            ) -> [np.ndarray]:
        """
        Perform the Harten correction to eliminate expansion shocks when using approximate riemann solvers.

        Parameters:
            - Wroe (RoePrimitiveState): Roe primitive state object, contains the roe average state calculated using
            the left and right conservative states used for the flux calculation.
            - WL (PrimitiveState): Left primitive state
            - WR (PrimitiveState): Right primitive state

        Returns:
            - lambda_roe_p (np.ndarray): Slow corrected wavespeeds
            - lambda_roe_m (np.ndarray): Fast corrected wavespeeds
        """

        # Right fast and slow wavespeeds
        lambda_R_p, lambda_R_m = self.wavespeeds_x(WR)
        # Left fast and slow wavespeeds
        lambda_L_p, lambda_L_m = self.wavespeeds_x(WL)
        # Roe fast and slow wavespeeds
        lambda_roe_p, lambda_roe_m = self.wavespeeds_x(Wroe)

        # Theta parameters for determining where to apply the correction
        theta_p = 2 * (lambda_R_p - lambda_L_p)
        theta_m = 2 * (lambda_R_m - lambda_L_m)

        # Prevent negative thetas
        theta_p = theta_p * (theta_p > 0)
        theta_m = theta_m * (theta_m > 0)

        # Corrected fast and slow Roe wavespeeds
        lambda_roe_p = np.where(np.absolute(lambda_roe_p) < theta_p,
                                0.5 * ((lambda_roe_p ** 2) / (theta_p + 1e-8) + theta_p),
                                lambda_roe_p)

        lambda_roe_m = np.where(np.absolute(lambda_roe_m) < theta_m,
                                0.5 * ((lambda_roe_m ** 2) / (theta_m + 1e-8) + theta_m),
                                lambda_roe_m)

        return lambda_roe_p, lambda_roe_m

    @abstractmethod
    def compute_flux(self, UL, UR):
        pass
