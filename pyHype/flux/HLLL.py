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
from pyHype.flux.base import FluxFunction
from pyHype.states.states import PrimitiveState, RoePrimitiveState, ConservativeState


class HLLL_FLUX_X(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self, UL, UR):
        pass


class HLLL_FLUX_Y(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self, UL, UR):
        pass