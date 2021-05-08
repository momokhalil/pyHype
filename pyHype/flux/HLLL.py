import numba
import numpy as np
from numba import float32
from pyHype.flux.base import FluxFunction
from pyHype.states.states import PrimitiveState, RoePrimitiveState, ConservativeState


class HLLL_FLUX_X(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self):
        pass


class HLLL_FLUX_Y(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self):
        pass