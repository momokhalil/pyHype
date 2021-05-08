import numpy as np
from pyHype.fvm.base import FiniteVolumeMethod


_ZERO_VEC = np.zeros((4, 1))

class SecondOrderLimited(FiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):
        super().__init__(inputs, global_nBLK)

    def reconstruct_state(self, U):
        limited_state   = self.flux_limiter.limit(U) * (U[8:] - U[:-8]) / 4

        self.UL.from_state_vector(U[:-4] + np.concatenate((_ZERO_VEC, limited_state), axis=0))
        self.UR.from_state_vector(U[4:] - np.concatenate((limited_state, _ZERO_VEC), axis=0))

    def shuffle(self):
        self.Flux_Y = self._shuffle.dot(self.Flux_Y)

    def get_flux(self, ref_BLK):

        for r in range(1, self.ny + 1):
            row = self.get_row(ref_BLK=ref_BLK, index=r)
            self.reconstruct_state(row)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[4 * self.nx * (r - 1):4 * self.nx * r] = flux[4:] - flux[:-4]

        for c in range(1, self.nx + 1):
            col = self.get_col(ref_BLK=ref_BLK, index=c)
            self.reconstruct_state(col)

            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            self.Flux_Y[4 * self.ny * (c - 1):4 * self.ny * c] = flux[4:] - flux[:-4]

        self.shuffle()
