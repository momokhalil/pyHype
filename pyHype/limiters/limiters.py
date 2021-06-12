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


class BarthJespersen(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return np.minimum(1, slope)


class Venkatakrishnan(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        s2 = slope ** 2
        return (s2 + 2 * slope) / (s2 + slope + 2)


class VanAlbada(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        s2 = slope ** 2
        return (s2 + slope) / (s2 + 1)


class VanLeer(SlopeLimiter):
    def __init__(self, inputs):
        super().__init__(inputs)

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        return (np.absolute(slope) + slope) / (slope + 1)
