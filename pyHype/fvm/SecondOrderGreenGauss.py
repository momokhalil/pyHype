import numpy as np
from pyHype.fvm.base import FiniteVolumeMethod
from pyHype.states import ConservativeState, PrimitiveState


_ZERO_VEC = np.zeros((4, 1))

class SecondOrderGreenGauss(FiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):
        super().__init__(inputs, global_nBLK)

        self.Ux = ConservativeState(inputs=self.inputs, size=self.nx + 2)
        self.Uy = ConservativeState(inputs=self.inputs, size=self.ny + 2)

    def get_flux(self, ref_BLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        for row in range(1, self.ny + 1):
            r = ref_BLK.fullrow(index=row)
            self.Ux.from_conservative_state_vector(r)

            self.reconstruct(self.Ux)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[4 * self.nx * (row - 1):4 * self.nx * row] = flux[4:] - flux[:-4]

        for col in range(1, self.nx + 1):
            c = ref_BLK.fullcol(index=col)
            self.Uy.from_conservative_state_vector(c)

            self.reconstruct(self.Uy)

            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            self.Flux_Y[4 * self.ny * (col - 1):4 * self.ny * col] = flux[4:] - flux[:-4]

        self.shuffle()

    def reconstruct(self, U: ConservativeState):

        if self.inputs.reconstruction_type == 'Primitive':
            stateL, stateR = self.reconstruct_state(U.to_primitive_state())
            self.UL.from_primitive_state_vector(stateL)
            self.UR.from_primitive_state_vector(stateR)

        elif self.inputs.reconstruction_type == 'Conservative':
            stateL, stateR = self.reconstruct_state(U)
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

    def reconstruct_state(self, state):
        limited_state   = self.flux_limiter.limit(state) * (state[8:] - state[:-8]) / 4
        stateL = state[:-4] + np.concatenate((_ZERO_VEC, limited_state), axis=0)
        stateR = state[4:] - np.concatenate((limited_state, _ZERO_VEC), axis=0)
        return stateL, stateR

    def shuffle(self):
        self.Flux_Y = self._shuffle.dot(self.Flux_Y)
