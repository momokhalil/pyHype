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

    def get_flux(self, refBLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Calculate x-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.nx + 1, 4))
        self.UR.reset(shape=(1, self.nx + 1, 4))

        # Iterate over all rows in block
        for row in range(self.ny):
            # Set x-direction U state vector holder to the rowth full-row of the block
            self.Ux.from_conservative_state_vector(refBLK.fullrow(row))
            # Reconstruct full-row
            self.reconstruct(self.Ux)
            # Calculate flux at each cell interface
            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            # Calculate flux difference between cell interfaces
            self.Flux_X[row, :, :] = (flux[4:] - flux[:-4]).reshape(-1, 4)

        # --------------------------------------------------------------------------------------------------------------
        # Calculate x-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.ny + 1, 4))
        self.UR.reset(shape=(1, self.ny + 1, 4))

        # Iterate over all columns in block
        for col in range(self.nx):
            # Set y-direction U state vector holder to the rowth full-row of the block
            self.Uy.from_conservative_state_vector(refBLK.fullcol(col))
            # Reconstruct full-column
            self.reconstruct(self.Uy)
            # Calculate flux at each cell interface
            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            # Calculate flux difference between cell interfaces
            self.Flux_Y[:, col, :] = (flux[4:] - flux[:-4]).reshape(-1, 4)


    def reconstruct_state(self, state) -> [np.ndarray]:
        limited_state   = self.flux_limiter.limit(state) * (state[:, 2:, :] - state[:, :-2, :]) / 4

        stateL = state[:, :-1, :] + np.concatenate((_ZERO_VEC, limited_state), axis=1)
        stateR = state[:, 1:, :] - np.concatenate((limited_state, _ZERO_VEC), axis=1)

        return stateL, stateR
