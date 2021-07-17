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
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from abc import abstractmethod

class SlopeLimiter:
    def __init__(self, inputs):
        self.inputs = inputs

    def get_slope(self,
                  state: np.ndarray,
                  ghostE: np.ndarray,
                  ghostW: np.ndarray,
                  ghostN: np.ndarray,
                  ghostS: np.ndarray,
                  quadE: np.ndarray,
                  quadW: np.ndarray,
                  quadN: np.ndarray,
                  quadS: np.ndarray,
                  ) -> [np.ndarray]:

        # initialise slopes
        sE = np.ones_like(state)
        sW = np.ones_like(state)
        sN = np.ones_like(state)
        sS = np.ones_like(state)

        # u_max for interior cells
        _east = np.concatenate((state[:, 1:], ghostE), axis=1)
        _west = np.concatenate((ghostW, state[:, :-1]), axis=1)
        _north = np.concatenate((state[1:, :], ghostN), axis=0)
        _south = np.concatenate((ghostS, state[:-1, :]), axis=0)

        u_max = np.maximum(state,
                           np.maximum(np.maximum(_east, _west),
                                      np.maximum(_north, _south))
                           )

        # u_min for interior cells
        u_min = np.minimum(state,
                           np.minimum(np.minimum(_east, _west),
                                      np.minimum(_north, _south))
                           )

        # Difference between largest/smallest value and average value
        dmax = u_max - state
        dmin = u_min - state

        # Difference between quadrature points and average value
        dE = quadE - state
        dW = quadW - state
        dN = quadN - state
        dS = quadS - state

        # Calculate slopes for each face

        # East face
        sE = np.where(dE > 0, self._compute_slope(dmax, dE), sE)
        sE = np.where(dE < 0, self._compute_slope(dmin, dE), sE)

        # West face
        sW = np.where(dW > 0, self._compute_slope(dmax, dW), sW)
        sW = np.where(dW < 0, self._compute_slope(dmin, dW), sW)

        # North face
        sN = np.where(dN > 0, self._compute_slope(dmax, dN), sN)
        sN = np.where(dN < 0, self._compute_slope(dmin, dN), sN)

        # South face
        sS = np.where(dS > 0, self._compute_slope(dmax, dS), sS)
        sS = np.where(dS < 0, self._compute_slope(dmin, dS), sS)

        return sE, sW, sN, sS

    def limit(self,
              state: np.ndarray,
              ghostE: np.ndarray,
              ghostW: np.ndarray,
              ghostN: np.ndarray,
              ghostS: np.ndarray,
              quadE: np.ndarray = None,
              quadW: np.ndarray = None,
              quadN: np.ndarray = None,
              quadS: np.ndarray = None,
              ) -> np.ndarray:

        # Calculate values needed to build slopes
        sE, sW, sN, sS = self.get_slope(state,
                                        ghostE, ghostW, ghostN, ghostS,
                                        quadE, quadW, quadN, quadS)

        # Get phi
        phiE = self._limiter_func(sE)
        phiW = self._limiter_func(sW)
        phiN = self._limiter_func(sN)
        phiS = self._limiter_func(sS)

        # Minimum limiter value
        phi = np.minimum(np.minimum(phiE, phiW), np.minimum(phiN, phiS))

        return np.where(phi < 0, 0, phi)

    @staticmethod
    def _compute_slope(diffminmax: np.ndarray,
                       diffquad: np.ndarray
                       ) -> np.ndarray:
        return diffminmax / (diffquad + 1e-8)

    @staticmethod
    @abstractmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        pass
