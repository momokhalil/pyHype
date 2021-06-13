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
import matplotlib.pyplot as plt
import numpy as np
from pyHype.fvm.base import MUSCLFiniteVolumeMethod
from pyHype.states import ConservativeState
import pyHype.utils.utils as utils

np.set_printoptions(precision=3)


class SecondOrderPWL(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        if inputs.nghost != 1:
            raise ValueError('Number of ghost cells must be equal to 1 for this method.')

        else:
            super().__init__(inputs, global_nBLK)

            self.Ux = ConservativeState(inputs=self.inputs, nx=self.nx + 2, ny=1)
            self.Uy = ConservativeState(inputs=self.inputs, nx=self.ny + 2, ny=1)

    @staticmethod
    def high_order_EW(refBLK):

        _ZERO_COL = np.zeros((refBLK.mesh.ny, 1, 4))

        high_ord_E = refBLK.gradx * (refBLK.mesh.EW_midpoint_x[:, 1:, :] - refBLK.mesh.x[:, :, np.newaxis]) \
                   + refBLK.grady * (refBLK.mesh.EW_midpoint_y[:, 1:, :] - refBLK.mesh.y[:, :, np.newaxis])

        high_ord_W = refBLK.gradx * (refBLK.mesh.EW_midpoint_x[:, :-1, :] - refBLK.mesh.x[:, :, np.newaxis]) \
                   + refBLK.grady * (refBLK.mesh.EW_midpoint_y[:, :-1, :] - refBLK.mesh.y[:, :, np.newaxis])

        return np.concatenate((_ZERO_COL, high_ord_E), axis=1), np.concatenate((high_ord_W, _ZERO_COL), axis=1)

    @staticmethod
    def high_order_NS(refBLK):

        _ZERO_ROW = np.zeros((1, refBLK.mesh.nx, 4))

        high_ord_N = refBLK.gradx * (refBLK.mesh.NS_midpoint_x[1:, :, :] - refBLK.mesh.x[:, :, np.newaxis]) \
                   + refBLK.grady * (refBLK.mesh.NS_midpoint_y[1:, :, :] - refBLK.mesh.y[:, :, np.newaxis])

        high_ord_S = refBLK.gradx * (refBLK.mesh.NS_midpoint_x[:-1, :, :] - refBLK.mesh.x[:, :, np.newaxis]) \
                   + refBLK.grady * (refBLK.mesh.NS_midpoint_y[:-1, :, :] - refBLK.mesh.y[:, :, np.newaxis])

        return np.concatenate((_ZERO_ROW, high_ord_N), axis=0), np.concatenate((high_ord_S, _ZERO_ROW), axis=0)


    def get_flux(self, refBLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        # Compute x and y direction gradients
        self.gradient(refBLK)

        # Get reconstructed quadrature points
        stateE, stateW, stateN, stateS = self.reconstruct(refBLK)

        # Calculate x-direction Flux
        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.nx + 1, 4))
        self.UR.reset(shape=(1, self.nx + 1, 4))

        # Iterate over all rows in block
        for row in range(self.ny):

            # Get vectors for the current row
            stateL = stateE[row:row+1, :, :]
            stateR = stateW[row:row+1, :, :]

            # Rotate to allign with coordinate axis
            utils.rotate_row(stateL, refBLK.mesh.thetax)
            utils.rotate_row(stateR, refBLK.mesh.thetax)

            # Set vectors based on left and right states
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

            # Calculate flux at each cell interface
            flux = self.flux_function_X.get_flux(self.UL, self.UR)
            utils.unrotate_row(flux, refBLK.mesh.thetax)

            # Calculate flux difference between cell interfaces
            self.Flux_X[row, :, :] = flux[:, 1:, :] - flux[:, :-1, :]

        # --------------------------------------------------------------------------------------------------------------
        # Calculate x-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.ny + 1, 4))
        self.UR.reset(shape=(1, self.ny + 1, 4))

        # Iterate over all columns in block
        for col in range(self.nx):

            # Get vectors for the current row
            stateL = stateN[:, col:col + 1, :].transpose((1, 0, 2))
            stateR = stateS[:, col:col + 1, :].transpose((1, 0, 2))

            # Rotate to allign with coordinate axis
            utils.rotate_row(stateL, refBLK.mesh.thetay)
            utils.rotate_row(stateR, refBLK.mesh.thetay)

            # Set vectors based on left and right states
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

            # Calculate flux at each cell interface
            flux = self.flux_function_Y.get_flux(self.UL, self.UR)
            utils.unrotate_row(flux, refBLK.mesh.thetay)

            # Calculate flux difference between cell interfaces
            self.Flux_Y[:, col, :] = (flux[:, 1:, :] - flux[:, :-1, :]).reshape(-1, 4)


    def reconstruct_state(self,
                          refBLK,
                          state: np.ndarray,
                          ghostE: np.ndarray,
                          ghostW: np.ndarray,
                          ghostN: np.ndarray,
                          ghostS: np.ndarray
                          ) -> [np.ndarray]:

        # East-West high order term for left and right states on each east-west cell interface
        high_ord_E, high_ord_W = self.high_order_EW(refBLK)

        # North-South high order term for left and right states on each north-south cell interface
        high_ord_N, high_ord_S = self.high_order_NS(refBLK)

        # East-West direction left and right states
        stateE = np.concatenate((ghostW, state), axis=1)
        stateW = np.concatenate((state, ghostE), axis=1)

        # North-South direction left and right states
        stateN = np.concatenate((ghostS, state), axis=0)
        stateS = np.concatenate((state, ghostN), axis=0)

        # Compute slope limiter
        phi = self.flux_limiter.limit(state,
                                      stateE, stateW, stateN, stateS,
                                      quadE=stateE + high_ord_E,
                                      quadW=stateW + high_ord_W,
                                      quadN=stateN + high_ord_N,
                                      quadS=stateS + high_ord_S)

        # Compute limited values at quadrature points
        stateE[:, 1:, :]  += phi * high_ord_E[:, 1:, :]
        stateW[:, :-1, :] += phi * high_ord_W[:, :-1, :]
        stateN[1:, :, :]  += phi * high_ord_N[1:, :, :]
        stateS[:-1, :, :] += phi * high_ord_S[:-1, :, :]

        return stateE, stateW, stateN, stateS
