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

import time

import matplotlib.pyplot as plt
import numpy as np
from pyHype.fvm.base import MUSCLFiniteVolumeMethod
from pyHype.states import ConservativeState
import pyHype.utils.utils as utils


_ZERO_VEC = np.zeros((1, 1, 4))

class SecondOrderGreenGauss(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        if inputs.nghost != 1:
            raise ValueError('Number of ghost cells must be equal to 1 for this method.')

        else:
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

        self.get_dUdx(refBLK)

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


    def reconstruct_state(self, state: np.ndarray) -> [np.ndarray]:
        limited_state   = self.flux_limiter.limit(state) * (state[:, 2:, :] - state[:, :-2, :]) / 4

        stateL = state[:, :-1, :] + np.concatenate((_ZERO_VEC, limited_state), axis=1)
        stateR = state[:, 1:, :] - np.concatenate((limited_state, _ZERO_VEC), axis=1)

        return stateL, stateR

    @staticmethod
    def get_interface_values_arithmetic(refBLK):

        # Concatenate mesh state and ghost block states
        catx = np.concatenate((refBLK.ghost.W.state.U, refBLK.state.U, refBLK.ghost.E.state.U), axis=1)
        caty = np.concatenate((refBLK.ghost.S.state.U, refBLK.state.U, refBLK.ghost.N.state.U), axis=0)

        # Compute arithmetic mean
        eastU = 0.5 * (catx[:, 1:-1, :] + catx[:, 2:, :])
        westU = 0.5 * (catx[:, :-2, :] + catx[:, 1:-1, :])
        northU = 0.5 * (caty[1:-1, :, :] + caty[2:, :, :])
        southU = 0.5 * (caty[:-2, :, :] + caty[1:-1, :, :])

        return eastU, westU, northU, southU

    def get_gradU(self, refBLK):

        # Concatenate mesh state and ghost block states
        interfaceE, interfaceW, interfaceN, interfaceS = self.get_interface_values_arithmetic(refBLK)

        # Calculate Side Length
        lengthE = refBLK.mesh.east_side_length()[:, :, np.newaxis]
        lengthW = refBLK.mesh.west_side_length()[:, :, np.newaxis]
        lengthN = refBLK.mesh.north_side_length()[:, :, np.newaxis]
        lengthS = refBLK.mesh.south_side_length()[:, :, np.newaxis]

        # Get each face's contribution to dUdx
        E = interfaceE * refBLK.mesh.EW_norm_x[0:, 1:,  np.newaxis] * lengthE
        W = interfaceW * refBLK.mesh.EW_norm_x[0:, :-1, np.newaxis] * lengthW * (-1)
        N = interfaceN * refBLK.mesh.NS_norm_x[1:, 0:,  np.newaxis] * lengthN
        S = interfaceS * refBLK.mesh.NS_norm_x[:-1, 0:, np.newaxis] * lengthS * (-1)

        # Compute dUdx
        dUdx = (E + W + N + S) / refBLK.mesh.A[:, :, np.newaxis]

        # Get each face's contribution to dUdy
        E = interfaceE * refBLK.mesh.EW_norm_y[0:, 1:,  np.newaxis] * lengthE
        W = interfaceW * refBLK.mesh.EW_norm_y[0:, :-1, np.newaxis] * lengthW * (-1)
        N = interfaceN * refBLK.mesh.NS_norm_y[1:, 0:,  np.newaxis] * lengthN
        S = interfaceS * refBLK.mesh.NS_norm_y[:-1, 0:, np.newaxis] * lengthS * (-1)

        # Compute dUdy
        dUdy = (E + W + N + S) / refBLK.mesh.A[:, :, np.newaxis]

        return dUdx, dUdy

