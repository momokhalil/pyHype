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

    def get_slope(self, U: np.ndarray, *args) -> np.ndarray:
        pass

    def limit(self, U: np.ndarray, *args) -> np.ndarray:
        pass

    @staticmethod
    def _limiter_func(slope: np.ndarray) -> np.ndarray:
        pass
