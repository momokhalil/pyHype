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
import numba as nb
from pyHype.states.states import ConservativeState, PrimitiveState, RoePrimitiveState
from profilehooks import profile


class FluxFunction:
    def __init__(self, inputs):
        self.inputs = inputs
        self.g = inputs.gamma
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.n = inputs.n

    def __call__(self,
                 WL: [PrimitiveState, np.ndarray] = None,
                 WR: [PrimitiveState, np.ndarray] = None,
                 UL: [ConservativeState, np.ndarray] = None,
                 UR: [ConservativeState, np.ndarray] = None,
                 *args, **kwargs):

        if UL is not None and UR is not None and (WL is None) and (WR is None):
            # Create Left and Right PrimitiveStates
            if isinstance(UL, ConservativeState):
                WL = UL.to_primitive_state()
            elif isinstance(UL, np.ndarray):
                WL = PrimitiveState(self.inputs, U_vector=UL)
            else:
                raise ValueError('Parameter UL must be of type ConservativeState or np.ndarray')

            if isinstance(UR, ConservativeState):
                WR = UR.to_primitive_state()
            elif isinstance(UR, np.ndarray):
                WR = PrimitiveState(self.inputs, U_vector=UR)
            else:
                raise ValueError('Parameter UR must be of type ConservativeState or np.ndarray')

        elif WL is not None and WR is not None and UL is None and UR is None:
            # Create Left and Right PrimitiveStates
            if isinstance(WL, np.ndarray):
                WL = PrimitiveState(self.inputs, W_vector=WL)
            else:
                raise ValueError('Parameter UL must be of type ConservativeState or np.ndarray')

            if isinstance(WR, np.ndarray):
                WR = PrimitiveState(self.inputs, W_vector=WR)
            else:
                raise ValueError('Parameter UR must be of type ConservativeState or np.ndarray')
        else:
            raise AttributeError('Only give UL and UR or WL and WR, not combination of U and W')

        return self.compute_flux(WL, WR, UL, UR)

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
            a = W.a_JIT(W.q3, W.q0, W.g)
            # Compute wavespeeds
            slow, fast = W.u - a, W.u + a
            return slow, fast
        else:
            raise TypeError('Input is not PrimitiveState.')


    def harten_correction_x(self,
                            Wroe: RoePrimitiveState,
                            WL: PrimitiveState,
                            WR: PrimitiveState,
                            Roe_p: np.ndarray = None,
                            Roe_m: np.ndarray = None,
                            L_p: np.ndarray = None,
                            L_m: np.ndarray = None,
                            R_p: np.ndarray = None,
                            R_m: np.ndarray = None,
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
        if R_p is None and R_m is None:
            R_p, R_m = self.wavespeeds_x(WR)
        # Left fast and slow wavespeeds
        if L_p is None and L_m is None:
            L_p, L_m = self.wavespeeds_x(WL)
        # Roe fast and slow wavespeeds
        if Roe_p is None and Roe_m is None:
            Roe_p, Roe_m = self.wavespeeds_x(Wroe)
        # Perform the harten correction using the JITed implementation
        _harten_correction_JIT(R_p, R_m, L_p, L_m, Roe_p, Roe_m)

        return Roe_p, Roe_m


    @staticmethod
    def _harten_correction_NUMPY(R_p, R_m, L_p, L_m, Roe_p, Roe_m):

        # Theta parameters for determining where to apply the correction
        theta_p = 2 * (R_p - L_p)
        theta_m = 2 * (R_m - L_m)

        # Prevent negative thetas
        theta_p = theta_p * (theta_p > 0)
        theta_m = theta_m * (theta_m > 0)

        # Corrected fast and slow Roe wavespeeds
        Roe_p = np.where(np.absolute(Roe_p) < theta_p,
                         0.5 * ((Roe_p ** 2) / (theta_p + 1e-8) + theta_p),
                         Roe_p)

        Roe_m = np.where(np.absolute(Roe_m) < theta_m,
                         0.5 * ((Roe_m ** 2) / (theta_m + 1e-8) + theta_m),
                         Roe_m)

        return Roe_p, Roe_m


    @abstractmethod
    def compute_flux(self,
                     WL: PrimitiveState,
                     WR: PrimitiveState,
                     UL: [ConservativeState, np.ndarray] = None,
                     UR: [ConservativeState, np.ndarray] = None,
                     ) -> np.ndarray:
        pass


@nb.njit(cache=True)
def _harten_correction_JIT(R_p, R_m, L_p, L_m, Roe_p, Roe_m):
    """
    Performs the harten correction on a given array of Roe wavespeeds, using the given theta indicator computed using
    Left and Right wavespeeds. The correction is applied in place due to the mutability of numpy arrays.

    Parameters:
        - Roe (np.ndarray): Numpy array holding the Roe wavespeeds which need to be corrected.
        - theta (np.ndarray): Numpy array holding the theta indicator used for determining if a correction is needed
    """

    # Loop through array
    for i in range(Roe_p.shape[0]):
        for j in range(Roe_p.shape[1]):
            # Save plus and minus thetas
            _tp = 2 * (R_p[i, j] - L_p[i, j])
            _tm = 2 * (R_m[i, j] - L_m[i, j])
            # Correct for zero or below zero values
            _tp = 1e-8 if _tp <= 0 else _tp
            _tm = 1e-8 if _tm <= 0 else _tm
            # Apply correction
            if np.absolute(Roe_p[i, j]) < _tp:
                Roe_p[i, j] = 0.5 * ((Roe_p[i, j] * Roe_p[i, j]) / _tp + _tp)
            if np.absolute(Roe_m[i, j]) < _tm:
                Roe_m[i, j] = 0.5 * ((Roe_m[i, j] * Roe_m[i, j]) / _tm + _tm)
