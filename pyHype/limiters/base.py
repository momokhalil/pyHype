import numba
import numpy as np
from numba import float32
import scipy.sparse as sparse
from numba import jit
from numba.experimental import jitclass
from pyHype.flux.base import FluxFunction
from pyHype.states.states import ConservativeState


class SlopeLimiter:
    def __init__(self, inputs):
        self.inputs = inputs

    def get_slope(self, U: np.ndarray) -> np.ndarray:
        pass

    def limit(self, U: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        pass
