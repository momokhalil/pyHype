import numba
import time
import numpy as np
from numba import float32
import scipy.sparse as sparse
from numba import jit
from numba.experimental import jitclass
from pyHype.limiters.base import SlopeLimiter


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
