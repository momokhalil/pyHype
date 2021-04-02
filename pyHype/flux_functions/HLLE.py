import numba
import numpy as np
from numba import float32
import scipy as sp
import scipy.sparse as sparse
from pyHype.flux_functions.base import FluxFunction
from pyHype.states import PrimitiveState, RoePrimitiveState, ConservativeState
from pyHype.utils import harten_correction_xdir, harten_correction_ydir


class HLLE_FLUX_X(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self):
        flux = np.empty((4 * self.inputs.nx + 4, 1))
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self.inputs, self.inputs.nx + 1, WL=WL, WR=WR)

        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)

        lambda_p = np.maximum(WL.u - WL.a(), Lp)
        lambda_m = np.maximum(WR.u - WR.a(), Lm)

        Fl = self._L.get_flux_X()
        Fr = self._R.get_flux_X()

        for i in range(1, self.inputs.nx + 1):
            l_p = lambda_p[i - 1]
            l_m = lambda_m[i - 1]

            if l_m >= 0:
                flux[4 * i - 4:4 * i] = Fl[4 * i - 4:4 * i]
            elif l_p <= 0:
                flux[4 * i - 4:4 * i] = Fr[4 * i - 4:4 * i]
            else:
                flux[4 * i - 4:4 * i] = l_p * Fl[4 * i - 4:4 * i] - l_m * Fr[4 * i - 4:4 * i] \
                                        + l_p * l_m * (self._R.U[4 * i - 4:4 * i] - self._L.U[4 * i - 4:4 * i]) / (
                                                    l_p - l_m)

        return flux


class HLLE_FLUX_Y(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

    def get_flux(self):
        flux = np.empty((4 * self.inputs.ny + 4, 1))
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self.inputs, self.inputs.nx + 1, WL=WL, WR=WR)

        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)

        lambda_p = np.maximum(WL.v - WL.a(), Lp)
        lambda_m = np.maximum(WR.v - WR.a(), Lm)

        Gl = self._L.get_flux_Y()
        Gr = self._R.get_flux_Y()

        for i in range(1, self.inputs.ny + 1):
            l_p = lambda_p[i - 1]
            l_m = lambda_m[i - 1]

            if l_m >= 0:
                flux[4 * i - 4:4 * i] = Gl[4 * i - 4:4 * i]
            elif l_p <= 0:
                flux[4 * i - 4:4 * i] = Gr[4 * i - 4:4 * i]
            else:
                flux[4 * i - 4:4 * i] = l_p * Gl[4 * i - 4:4 * i] - l_m * Gr[4 * i - 4:4 * i] \
                                        + l_p * l_m * (self._R.U[4 * i - 4:4 * i] - self._L.U[4 * i - 4:4 * i]) / (
                                                    l_p - l_m)
        return flux
