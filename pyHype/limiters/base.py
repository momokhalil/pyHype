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
import numba as nb

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
        _EW = np.concatenate((ghostW, state, ghostE), axis=1)
        _NS = np.concatenate((ghostS, state, ghostN), axis=0)
        _vals = (state, _EW[:, :-2], _EW[:, 2:], _NS[:-2, :], _NS[2:, :])

        # Difference between largest/smallest value and average value
        dmax = np.maximum.reduce(_vals) - state
        dmin = np.minimum.reduce(_vals) - state

        # Difference between quadrature points and average value
        dE = quadE - state
        dW = quadW - state
        dN = quadN - state
        dS = quadS - state

        # Calculate slopes for each face

        # East face
        sE = np.where(dE > 0, self._compute_slope(dmax, dE), sE)
        sE = np.where(dE < 0, self._compute_slope(dmin, dE), sE)
        #sE = _compute_slope(dmax, dmin, dE)

        # West face
        sW = np.where(dW > 0, self._compute_slope(dmax, dW), sW)
        sW = np.where(dW < 0, self._compute_slope(dmin, dW), sW)
        #sW = _compute_slope(dmax, dmin, dW)

        # North face
        sN = np.where(dN > 0, self._compute_slope(dmax, dN), sN)
        sN = np.where(dN < 0, self._compute_slope(dmin, dN), sN)
        #sN = _compute_slope(dmax, dmin, dN)

        # South face
        sS = np.where(dS > 0, self._compute_slope(dmax, dS), sS)
        sS = np.where(dS < 0, self._compute_slope(dmin, dS), sS)
        #sS = _compute_slope(dmax, dmin, dS)

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

        # Minimum limiter value
        phi = np.minimum.reduce((self._limiter_func(sE),
                                 self._limiter_func(sW),
                                 self._limiter_func(sN),
                                 self._limiter_func(sS)))

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

@nb.vectorize(nopython=True)
def _compute_slope(dmax, dmin, dU):
    if dU > 0:      return dmax / (dU + 1e-8)
    elif dU < 0:    return dmin / (dU + 1e-8)
    else:           return 1

