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

np.set_printoptions(precision=3)


_ZERO_VEC = np.zeros((1, 1, 4))


"""
------------------------------------------------------------------------------------------------------------------------

                                                IMPORTANT

This method is experimental and still under development. DO NOT USE FOR SIMULATION UNTIL THIS MESSAGE IS REMOVED.

* This method is not optimized
* This method does not follow standard architecture
* This method is still being debugged
* This method still lacks a few features (state rotation) to be fully compatible with non-cartesian axes aligned geom.
------------------------------------------------------------------------------------------------------------------------
"""

class SecondOrderGreenGauss(MUSCLFiniteVolumeMethod):
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

        state_E, state_W, state_N, state_S = self.reconstruct_state(refBLK)
        # --------------------------------------------------------------------------------------------------------------
        # Calculate x-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.nx + 1, 4))
        self.UR.reset(shape=(1, self.nx + 1, 4))

        # Iterate over all rows in block
        for row in range(self.ny):

            # Get vectors for the current row
            stateL = state_E[row:row+1, :, :]
            stateR = state_W[row:row+1, :, :]

            # Rotate to allign with coordinate axis
            #rot_stateL = utils.rotate_row(stateL, refBLK.mesh.thetax)
            #rot_stateR = utils.rotate_row(stateR, refBLK.mesh.thetax)

            # Set vectors based on left and right states
            #self.UL.from_conservative_state_vector(rot_stateL)
            #self.UR.from_conservative_state_vector(rot_stateR)

            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

            # Calculate flux at each cell interface
            #rot_flux = self.flux_function_X.get_flux(self.UL, self.UR)
            #flux = utils.unrotate_row(rot_flux.reshape(1, -1, 4), refBLK.mesh.thetax)

            flux = self.flux_function_X.get_flux(self.UL, self.UR).reshape(1, -1, 4)
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
            stateL = state_N[:, col:col+1, :].transpose((1, 0, 2))
            stateR = state_S[:, col:col+1, :].transpose((1, 0, 2))

            # Set vectors based on left and right states
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

            # Calculate flux at each cell interface
            flux = self.flux_function_Y.get_flux(self.UL, self.UR).reshape(-1, 4)

            # Calculate flux difference between cell interfaces
            self.Flux_Y[:, col, :] = (flux[1:, :] - flux[:-1, :])


    def reconstruct_state(self, refBLK) -> [np.ndarray]:

        _ZR = np.zeros((1, refBLK.mesh.nx, 4))
        _ZC = np.zeros((refBLK.mesh.ny, 1, 4))

        # Compute x and y direction gradients
        self.compute_grad(refBLK)

        # East-West high order term for left and right states on each east-west cell interface
        high_ord_E, high_ord_W = self.high_order_EW(refBLK)

        # North-South high order term for left and right states on each north-south cell interface
        high_ord_N, high_ord_S = self.high_order_NS(refBLK)

        # East-West direction left and right states
        state_E = np.concatenate((refBLK.ghost.W.state.U, refBLK.state.U), axis=1)
        state_W = np.concatenate((refBLK.state.U, refBLK.ghost.E.state.U), axis=1)

        # North-South direction left and right states
        state_N = np.concatenate((refBLK.ghost.S.state.U, refBLK.state.U), axis=0)
        state_S = np.concatenate((refBLK.state.U, refBLK.ghost.N.state.U), axis=0)

        # Values at east, west, north, south quadrature points
        quad_E = state_E + high_ord_E
        quad_W = state_W + high_ord_W

        quad_N = state_N + high_ord_N
        quad_S = state_S + high_ord_S

        # Compute slope limiter
        phi = self.flux_limiter.limit(refBLK.state.U, state_E, state_W, state_N, state_S,
                                      quad_E, quad_W, quad_N, quad_S)

        # Compute limited values at quadrature points
        quad_E = state_E + np.concatenate((_ZC, phi), axis=1) * high_ord_E
        quad_W = state_W + np.concatenate((phi, _ZC), axis=1) * high_ord_W
        quad_N = state_N + np.concatenate((phi, _ZR), axis=0) * high_ord_N
        quad_S = state_S + np.concatenate((_ZR, phi), axis=0) * high_ord_S

        return quad_E, quad_W, quad_N, quad_S

    def compute_grad(self, refBLK):

        # Concatenate mesh state and ghost block states
        interfaceEW, interfaceNS = self.get_interface_values(refBLK)

        # Calculate Side Length
        lengthE = refBLK.mesh.east_side_length()[:, :, np.newaxis]
        lengthW = refBLK.mesh.west_side_length()[:, :, np.newaxis]
        lengthN = refBLK.mesh.north_side_length()[:, :, np.newaxis]
        lengthS = refBLK.mesh.south_side_length()[:, :, np.newaxis]

        # Get each face's contribution to dUdx
        E = interfaceEW[:, 1:, :]  * refBLK.mesh.EW_norm_x[0:, 1:,  np.newaxis] * lengthE
        W = interfaceEW[:, :-1, :] * refBLK.mesh.EW_norm_x[0:, :-1, np.newaxis] * lengthW * (-1)
        N = interfaceNS[1:, :, :]  * refBLK.mesh.NS_norm_x[1:, 0:,  np.newaxis] * lengthN
        S = interfaceNS[:-1, :, :] * refBLK.mesh.NS_norm_x[:-1, 0:, np.newaxis] * lengthS * (-1)

        # Compute dUdx
        refBLK.gradx = (E + W + N + S) / refBLK.mesh.A[:, :, np.newaxis]

        # Get each face's contribution to dUdy
        E = interfaceEW[:, 1:, :]  * refBLK.mesh.EW_norm_y[0:, 1:,  np.newaxis] * lengthE
        W = interfaceEW[:, :-1, :] * refBLK.mesh.EW_norm_y[0:, :-1, np.newaxis] * lengthW * (-1)
        N = interfaceNS[1:, :, :]  * refBLK.mesh.NS_norm_y[1:, 0:,  np.newaxis] * lengthN
        S = interfaceNS[:-1, :, :] * refBLK.mesh.NS_norm_y[:-1, 0:, np.newaxis] * lengthS * (-1)

        # Compute dUdy
        refBLK.grady = (E + W + N + S) / refBLK.mesh.A[:, :, np.newaxis]

        #plt.contourf(refBLK.mesh.x, refBLK.mesh.y, (refBLK.grady + refBLK.gradx)[:, :, 0], 50, cmap='magma')
        #plt.contourf(refBLK.mesh.x, refBLK.mesh.y, (N + S)[:, :, 0], 50)
        #plt.pause(0.1)
