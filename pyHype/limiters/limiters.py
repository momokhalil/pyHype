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


class VanAlbada(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_slope(self, U: np.ndarray) -> np.ndarray:
        slope = (U[:, 2:, :] - U[:, 1:-1, :]) / (U[:, 1:-1, :] - U[:, :-2, :] + 1e-8)
        return slope * (slope > 0)

    def limit(self, U: np.ndarray) -> np.ndarray:
        slope = self.get_slope(U)
        return 2 * (self._limiter_func(slope)) / (slope + 1)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.square(slope) + slope) / (np.square(slope) + 1)


class VanLeer(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_slope(self, U: np.ndarray) -> np.ndarray:
        slope = (U[8:] - U[4:-4]) / (U[4:-4] - U[:-8] + 1e-8)
        return slope * (slope > 0)

    def limit(self, U: np.ndarray) -> np.ndarray:
        slope = self.get_slope(U)
        return 2 * (self._limiter_func(slope)) / (slope + 1)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.absolute(slope) + slope) / (slope + 1)
