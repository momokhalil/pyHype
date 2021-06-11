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
from pyHype.limiters import SlopeLimiter


class Venkatakrishnan(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_slope(self,
                  state: np.ndarray,
                  UE: np.ndarray = None,
                  UW: np.ndarray = None,
                  UN: np.ndarray = None,
                  US: np.ndarray = None,
                  quadE: np.ndarray = None,
                  quadW: np.ndarray = None,
                  quadN: np.ndarray = None,
                  quadS: np.ndarray = None,
                  ) -> [np.ndarray]:

        # u_max for interior cells
        u_max = np.maximum(state,
                           np.maximum(np.maximum(UE[:, :-1, :], UW[:, 1:, :]),
                                      np.maximum(UN[:-1, :, :], US[1:, :, :]))
                           )

        # u_min for interior cells
        u_min = np.minimum(state,
                           np.minimum(np.minimum(UE[:, :-1, :], UW[:, 1:, :]),
                                      np.minimum(UN[:-1, :, :], US[1:, :, :]))
                           )

        # Difference between quadrature point and cell average
        diff_E = quadE[:, 1:, :] - state
        diff_W = quadW[:, :-1, :] - state
        diff_N = quadN[1:, :, :] - state
        diff_S = quadS[:-1, :, :] - state

        # Difference between min/max and cell average
        diff_max = u_max - state
        diff_min = u_min - state

        return diff_max, diff_min, diff_E, diff_W, diff_N, diff_S

    @staticmethod
    def _compute_slope(diffminmax: np.ndarray,
                       diffquad: np.ndarray
                       ) -> np.ndarray:
        return diffminmax / (diffquad + 1e-8)

    def limit(self, 
              state: np.ndarray,
              UE: np.ndarray = None,
              UW: np.ndarray = None,
              UN: np.ndarray = None,
              US: np.ndarray = None,
              quadE: np.ndarray = None,
              quadW: np.ndarray = None,
              quadN: np.ndarray = None,
              quadS: np.ndarray = None,
              ) -> np.ndarray:

        phiE = np.ones_like(state)
        phiW = np.ones_like(state)
        phiN = np.ones_like(state)
        phiS = np.ones_like(state)

        # Calculate values needed to build slopes
        dmax, dmin, dE, dW, dN, dS = self.get_slope(state, UE, UW, UN, US, quadE, quadW, quadN, quadS)

        # Get phi

        # East face
        maxE = self._compute_slope(dmax, dE)                        # Maximum slope
        minE = self._compute_slope(dmin, dE)                        # Minimum slope
        phiE = np.where(dE > 0, self._limiter_func(maxE), phiE)     # Limiter for max slope when uE - ui > 0
        phiE = np.where(dE < 0, self._limiter_func(minE), phiE)     # Limiter for max slope when uE - ui < 0

        # West face
        maxW = self._compute_slope(dmax, dW)
        minW = self._compute_slope(dmin, dW)
        phiW = np.where(dW > 0, self._limiter_func(maxW), phiW)
        phiW = np.where(dW < 0, self._limiter_func(minW), phiW)

        # North face
        maxN = self._compute_slope(dmax, dN)
        minN = self._compute_slope(dmin, dN)
        phiN = np.where(dN > 0, self._limiter_func(maxN), phiN)
        phiN = np.where(dN < 0, self._limiter_func(minN), phiN)

        # South face
        maxS = self._compute_slope(dmax, dS)
        minS = self._compute_slope(dmin, dS)
        phiS = np.where(dS > 0, self._limiter_func(maxS), phiS)
        phiS = np.where(dS < 0, self._limiter_func(minS), phiS)

        # Minimum limiter value
        phi = np.minimum(np.minimum(phiE, phiW), np.minimum(phiN, phiS))

        return np.where(phi < 0, 0, phi)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        s2 = slope ** 2
        return (s2 + 2 * slope) / (s2 + slope + 2)


class VanAlbada(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_slope(self, U: np.ndarray, *args) -> np.ndarray:
        slope = (U[:, 2:, :] - U[:, 1:-1, :]) / (U[:, 1:-1, :] - U[:, :-2, :] + 1e-8)
        return slope * (slope > 0)

    def limit(self, U: np.ndarray, *args) -> np.ndarray:
        slope = self.get_slope(U)
        return 2 * (self._limiter_func(slope)) / (slope + 1)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.square(slope) + slope) / (np.square(slope) + 1)


class VanLeer(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_slope(self, U: np.ndarray, *args) -> np.ndarray:
        slope = (U[8:] - U[4:-4]) / (U[4:-4] - U[:-8] + 1e-8)
        return slope * (slope > 0)

    def limit(self, U: np.ndarray, *args) -> np.ndarray:
        slope = self.get_slope(U)
        return 2 * (self._limiter_func(slope)) / (slope + 1)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.absolute(slope) + slope) / (slope + 1)
