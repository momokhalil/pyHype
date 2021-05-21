import numpy as np
import time
from pyHype.fvm.base import MUSCLFiniteVolumeMethod
from pyHype.states import ConservativeState


_ZERO_VEC = np.zeros((1, 1, 4))

class LeastSquaresSecondOrder(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        super().__init__(inputs, global_nBLK)

        self.Ux = ConservativeState(inputs=self.inputs, nx=self.nx + 2, ny=1)
        self.Uy = ConservativeState(inputs=self.inputs, nx=self.ny + 2, ny=1)

    def get_flux(self, ref_BLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        for row in range(self.ny):
            row_state = ref_BLK.fullrow(row)
            self.Ux.from_conservative_state_vector(row_state)

            self.reconstruct(self.Ux)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[row, :, :] = (flux[4:] - flux[:-4]).reshape(-1, 4)

        for col in range(self.nx):
            col_state = ref_BLK.fullcol(col)
            self.Uy.from_conservative_state_vector(col_state)

            self.reconstruct(self.Uy)

            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            self.Flux_Y[:, col, :] = (flux[4:] - flux[:-4]).reshape(-1, 4)


    def reconstruct(self, U: ConservativeState):

        if self.inputs.reconstruction_type == 'Primitive':
            stateL, stateR = self.reconstruct_state(U.to_primitive_state())
            self.UL.from_primitive_state_vector(stateL)
            self.UR.from_primitive_state_vector(stateR)

        elif self.inputs.reconstruction_type == 'Conservative':
            stateL, stateR = self.reconstruct_state(U)
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

    def reconstruct_primitive(self, U: ConservativeState):
        pass

    def reconstruct_conservative(self, U: ConservativeState):
        pass

    def reconstruct_state(self, state):
        limited_state   = self.flux_limiter.limit(state) * (state[:, 2:, :] - state[:, :-2, :]) / 4

        stateL = state[:, :-1, :] + np.concatenate((_ZERO_VEC, limited_state), axis=1)
        stateR = state[:, 1:, :] - np.concatenate((limited_state, _ZERO_VEC), axis=1)
        return stateL, stateR

    def get_dWdx(self):
        pass

    def get_dWdy(self):
        pass

    def get_dUdx(self):
        pass

    def get_dUdy(self):
        pass