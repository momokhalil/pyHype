import numba
import numpy as np
from numba import float32
import scipy as sp
import scipy.sparse as sparse
from pyHype.flux_functions.base import FluxFunction
from pyHype.states import PrimitiveState, RoePrimitiveState, ConservativeState
from pyHype.utils import harten_correction_xdir, harten_correction_ydir


class HLLL_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass


class HLLL_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass