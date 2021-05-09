import numpy as np
from pyHype.fvm.base import FiniteVolumeMethod
from pyHype.states import ConservativeState, PrimitiveState, State


_ZERO_VEC = np.zeros((4, 1))

class SecondOrderLimited(FiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):
        super().__init__(inputs, global_nBLK)

    def get_flux(self, ref_BLK):

        Ux = ConservativeState(inputs=self.inputs, size_=self.nx + 2)
        Uy = ConservativeState(inputs=self.inputs, size_=self.ny + 2)

        for row in range(1, self.ny + 1):
            Ux.from_state_vector(ref_BLK.fullrow(index=row))

            self.reconstruct(Ux)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[4 * self.nx * (row - 1):4 * self.nx * row] = flux[4:] - flux[:-4]

        for col in range(1, self.nx + 1):
            Uy.from_state_vector(ref_BLK.fullcol(index=col))

            self.reconstruct(Uy)

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
            return instate.to_W()
        elif self.inputs.reconstruction_type == 'Conservative':
            return instate
        else:
            print('WTF')

    def postprocessing(self, stateL, stateR):

        if self.inputs.reconstruction_type == 'Primitive':
            W = PrimitiveState(inputs=self.inputs, size_=int(stateL.shape[0]/4))

            W.update(stateL)
            self.UL = W.to_U()

            W.update(stateR)
            self.UR = W.to_U()

        elif self.inputs.reconstruction_type == 'Conservative':
            self.UL.from_state_vector(stateL)
            self.UR.from_state_vector(stateR)

        else:
            print('WTF')

    def shuffle(self):
        self.Flux_Y = self._shuffle.dot(self.Flux_Y)
