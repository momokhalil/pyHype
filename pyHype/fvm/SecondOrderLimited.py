import numpy as np
from numba.experimental import jitclass
from pyHype.fvm.base import FiniteVolumeMethod
from pyHype.states.states import ConservativeState


class SecondOrderLimited(FiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        super().__init__(inputs, global_nBLK)
        self._slope = None

    def get_slope(self, U):
        slope = (U[8:] - U[4:-4]) / (U[4:-4] - U[:-8] + 1e-8)
        return slope * (slope > 0)

    def get_limiter(self, U):
        slope = self.get_slope(U)
        return 2 * (self._flux_limiter(slope)) / (slope + 1)

    def reconstruct_state_X(self, U):
        limited_state = self.get_limiter(U) * (U[8:] - U[:-8])
        left, right = U[:-4], U[4:]
        left[4:] += limited_state / 4
        right[:-4] -= limited_state / 4

        UL = ConservativeState(self.inputs, self.nx + 1)
        UL.from_state_vector(left)

        UR = ConservativeState(self.inputs, self.nx + 1)
        UR.from_state_vector(right)

        self._flux_function_X.set_left_state(UL)
        self._flux_function_X.set_right_state(UR)

    def reconstruct_state_Y(self, U):
        limited_state   = self.get_limiter(U) * (U[8:] - U[:-8])
        left, right     = U[:-4], U[4:]
        left[4:]        += limited_state
        right[:-4]      -= limited_state

        UL = ConservativeState(self.inputs, self.ny + 1)
        UL.from_state_vector(left)

        UR = ConservativeState(self.inputs, self.ny + 1)
        UR.from_state_vector(right)

        self._flux_function_Y.set_left_state(UL)
        self._flux_function_Y.set_right_state(UR)

    def get_flux(self, ref_BLK):

        for r in range(1, self.ny + 1):
            row = self.get_row(ref_BLK=ref_BLK, index=r)
            self.reconstruct_state_X(row)

            flux = self._flux_function_X.get_flux()
            self.Flux_X[4 * self.nx * (r - 1):4 * self.nx * r] = flux[4:] - flux[:-4]

        for c in range(1, self.nx + 1):
            col = self.get_col(ref_BLK=ref_BLK, index=c)
            self.reconstruct_state_Y(col)

            flux = self._flux_function_Y.get_flux()
            self.Flux_Y[4 * self.ny * (c - 1):4 * self.ny * c] = flux[4:] - flux[:-4]

        self.shuffle()

    @staticmethod
    def get_row(ref_BLK, index: int) -> np.ndarray:
        return np.vstack((ref_BLK.boundary_blocks.W[index],
                          ref_BLK.row(index),
                          ref_BLK.boundary_blocks.E[index]))

    @staticmethod
    def get_col(ref_BLK, index: int) -> np.ndarray:
        return np.vstack((ref_BLK.boundary_blocks.S[index],
                          ref_BLK.col(index),
                          ref_BLK.boundary_blocks.N[index]))

    def shuffle(self):
        self.Flux_Y = self._shuffle.dot(self.Flux_Y)
