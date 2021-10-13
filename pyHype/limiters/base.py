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
import numba as nb
from abc import abstractmethod
from profilehooks import profile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock

class SlopeLimiter:
    def __init__(self, inputs):
        self.inputs = inputs
        self.sE = np.zeros((inputs.ny, inputs.nx, 1))
        self.sW = np.zeros((inputs.ny, inputs.nx, 1))
        self.sN = np.zeros((inputs.ny, inputs.nx, 1))
        self.sS = np.zeros((inputs.ny, inputs.nx, 1))

    def get_slope(self,
                  refBLK: QuadBlock,
                  quadE: np.ndarray,
                  quadW: np.ndarray,
                  quadN: np.ndarray,
                  quadS: np.ndarray,
                  ) -> [np.ndarray]:

        _EW = np.concatenate((refBLK.ghost.W.state.Q, refBLK.state.Q, refBLK.ghost.E.state.Q), axis=1)
        _NS = np.concatenate((refBLK.ghost.S.state.Q, refBLK.state.Q, refBLK.ghost.N.state.Q), axis=0)
        # Values for min/max evaluation
        _vals = (refBLK.state.Q, _EW[:, :-2], _EW[:, 2:], _NS[:-2, :], _NS[2:, :])
        # Difference between largest/smallest value and average value
        dmax = np.maximum.reduce(_vals) - refBLK.state.Q
        dmin = np.minimum.reduce(_vals) - refBLK.state.Q
        # Difference between quadrature points and average value
        dE = quadE - refBLK.state.Q
        dW = quadW - refBLK.state.Q
        dN = quadN - refBLK.state.Q
        dS = quadS - refBLK.state.Q
        # Calculate slopes for each face
        sE = self._compute_slope(dmax, dmin, dE)
        sW = self._compute_slope(dmax, dmin, dW)
        sN = self._compute_slope(dmax, dmin, dN)
        sS = self._compute_slope(dmax, dmin, dS)
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
    def _compute_slope(dmax, dmin, dE):
        _s = np.ones_like(dE)
        for i in range(dE.shape[0]):
            for j in range(dE.shape[1]):
                for v in range(4):
                    if dE[i, j, v] > 0:
                        _s[i, j, v] = dmax[i, j, v] / (dE[i, j, v] + 1e-8)
                    elif dE[i, j, v] < 0:
                        _s[i, j, v] = dmin[i, j, v] / (dE[i, j, v] + 1e-8)
        return _s


    @staticmethod
    @abstractmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        pass
