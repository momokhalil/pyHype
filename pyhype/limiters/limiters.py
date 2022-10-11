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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import numba as nb
from pyhype.limiters import SlopeLimiter


class BarthJespersen(SlopeLimiter):
    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return np.minimum(1, slope)


class Venkatakrishnan(SlopeLimiter):
    def _limiter_func(self, slope: np.ndarray) -> np.ndarray:
        if self.inputs.use_JIT:
            return self._venkata(slope)
        s2 = slope**2
        return (s2 + 2 * slope) / (s2 + slope + 2)

    @staticmethod
    @nb.njit(cache=True)
    def _venkata(slope: np.ndarray):
        s = np.zeros_like(slope)
        _s = 0.0
        for i in range(slope.shape[0]):
            for j in range(slope.shape[1]):
                for k in range(slope.shape[2]):
                    _s = slope[i, j, k]
                    s2 = _s * _s
                    s[i, j, k] = (s2 + 2 * _s) / (s2 + _s + 2)
        return s


class VanAlbada(SlopeLimiter):
    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        s2 = slope**2
        return (s2 + slope) / (s2 + 1)


class VanLeer(SlopeLimiter):
    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.absolute(slope) + slope) / (slope + 1)
