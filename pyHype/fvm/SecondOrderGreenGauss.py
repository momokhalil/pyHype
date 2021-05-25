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

import numpy as np
import time
from pyHype.fvm.base import MUSCLFiniteVolumeMethod
from pyHype.states import ConservativeState


_ZERO_VEC = np.zeros((1, 1, 4))

class SecondOrderGreenGauss(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        super().__init__(inputs, global_nBLK)

        self.Ux = ConservativeState(inputs=self.inputs, nx=self.nx + 2, ny=1)
        self.Uy = ConservativeState(inputs=self.inputs, nx=self.ny + 2, ny=1)

    def get_flux(self, ref_BLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        #self.UL.reset(shape=(1, self.nx + 1, 4))
        #self.UR.reset(shape=(1, self.nx + 1, 4))

        for row in range(self.ny):
            row_state = ref_BLK.fullrow(row)
            self.Ux.from_conservative_state_vector(row_state)

            self.reconstruct(self.Ux)

            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            self.Flux_X[row, :, :] = (flux[4:] - flux[:-4]).reshape(-1, 4)

        #self.UL.reset(shape=(1, self.ny + 1, 4))
        #self.UR.reset(shape=(1, self.ny + 1, 4))

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

    def reconstruct_state(self, state):
        limited_state   = self.flux_limiter.limit(state) * (state[:, 2:, :] - state[:, :-2, :]) / 4

        stateL = state[:, :-1, :] + np.concatenate((_ZERO_VEC, limited_state), axis=1)
        stateR = state[:, 1:, :] - np.concatenate((limited_state, _ZERO_VEC), axis=1)
        return stateL, stateR
