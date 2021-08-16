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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock

class SlopeLimiter:
    def __init__(self, inputs):
        self.inputs = inputs

    def get_slope(self,
                  refBLK: QuadBlock,
                  quadE: np.ndarray,
                  quadW: np.ndarray,
                  quadN: np.ndarray,
                  quadS: np.ndarray,
                  ) -> [np.ndarray]:

        # State
        _state = refBLK.state.Q

        # u_max for interior cells
        _EW = np.concatenate((refBLK.ghost.W.state.Q,
                              _state,
                              refBLK.ghost.E.state.Q),
                             axis=1)
        # u_min for interior cells
        _NS = np.concatenate((refBLK.ghost.S.state.Q,
                              _state,
                              refBLK.ghost.N.state.Q),
                             axis=0)
        # Values for min/max evaluation
        _vals = (_state, _EW[:, :-2], _EW[:, 2:], _NS[:-2, :], _NS[2:, :])

        # Difference between largest/smallest value and average value
        dmax = np.maximum.reduce(_vals) - _state
        dmin = np.minimum.reduce(_vals) - _state

        # Difference between quadrature points and average value
        dE = quadE - _state
        dW = quadW - _state
        dN = quadN - _state
        dS = quadS - _state

        # Calculate slopes for each face
        sE = self.__compute_slope(dmax, dmin, dE)
        sW = self.__compute_slope(dmax, dmin, dW)
        sN = self.__compute_slope(dmax, dmin, dN)
        sS = self.__compute_slope(dmax, dmin, dS)

        return sE, sW, sN, sS

    def limit(self,
              refBLK: QuadBlock,
              quadE: np.ndarray = None,
              quadW: np.ndarray = None,
              quadN: np.ndarray = None,
              quadS: np.ndarray = None,
              ) -> np.ndarray:

        # Calculate values needed to build slopes
        sE, sW, sN, sS = self.get_slope(refBLK, quadE, quadW, quadN, quadS)

        # Minimum limiter value
        phi = np.minimum.reduce((self._limiter_func(sE),
                                 self._limiter_func(sW),
                                 self._limiter_func(sN),
                                 self._limiter_func(sS)))

        return np.where(phi < 0, 0, phi)

    @staticmethod
    @nb.njit(cache=True)
    def _compute_slope(dmax, dmin, dU):
        _s = np.ones_like(dU)
        for i in range(dU.shape[0]):
            for j in range(dU.shape[1]):
                for v in range(4):
                    if dU[i, j, v] > 0:
                        _s[i, j, v] = dmax[i, j, v] / (dU[i, j, v] + 1e-8)
                    elif dU[i, j, v] < 0:
                        _s[i, j, v] = dmin[i, j, v] / (dU[i, j, v] + 1e-8)
        return _s

    def __compute_slope(self, dmax, dmin, dU):
        _s = np.ones_like(dU)
        _s = np.where(dU > 0, dmax / (dU + 1e-8), _s)
        _s = np.where(dU < 0, dmin / (dU + 1e-8), _s)
        return _s

    @staticmethod
    @abstractmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        pass
