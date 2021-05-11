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
            row = ref_BLK.fullrow(index=row)
            self.Ux.from_conservative_state_vector(row)

            self.reconstruct(self.Ux)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[4 * self.nx * (row - 1):4 * self.nx * row] = flux[4:] - flux[:-4]

        for col in range(1, self.nx + 1):
            col = ref_BLK.fullcol(index=col)
            self.Uy.from_conservative_state_vector(col)

            self.reconstruct(self.Uy)

            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            self.Flux_Y[4 * self.ny * (col - 1):4 * self.ny * col] = flux[4:] - flux[:-4]

        self.shuffle()

    def reconstruct_state(self, state):
        limited_state   = self.flux_limiter.limit(state) * (state[8:] - state[:-8]) / 4
        stateL = state[:-4] + np.concatenate((_ZERO_VEC, limited_state), axis=0)
        stateR = state[4:] - np.concatenate((limited_state, _ZERO_VEC), axis=0)
        return stateL, stateR

    def reconstruct(self, U: ConservativeState):
        state = self.preprocessing(instate=U)
        stateL, stateR = self.reconstruct_state(state)
        self.postprocessing(stateL, stateR)

    def preprocessing(self, instate: ConservativeState):
        if self.inputs.reconstruction_type == 'Primitive':
            return instate.to_primitive_state()
        elif self.inputs.reconstruction_type == 'Conservative':
            return instate
        else:
            print('WTF')

    def postprocessing(self, stateL, stateR):
        if self.inputs.reconstruction_type == 'Primitive':
            self.UL.from_primitive_state_vector(stateL)
            self.UR.from_primitive_state_vector(stateR)

        elif self.inputs.reconstruction_type == 'Conservative':
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

        else:
            print('WTF')

    def shuffle(self):
        self.Flux_Y = self._shuffle.dot(self.Flux_Y)
