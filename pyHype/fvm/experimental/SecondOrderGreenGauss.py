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

            """if col == 5:
                print(refBLK.state.U[:, col, 0])
                print(self.UL.rho)
                print(self.UR.rho)
                print(flux)

                plt.pause(5)"""

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

        """print('-------------------------------------------')
        print(refBLK.ghost.S.state.U[:, :, 0])
        print(refBLK.ghost.N.state.U[:, :, 0])
        print('-------------------------------------------')
        print(state_N[:, :, 0])
        print(state_S[:, :, 0])

        plt.pause(10)"""

        """plt.contourf(refBLK.mesh.x, refBLK.mesh.y, state_N[1:, :, 0], 50, cmap='magma')
        plt.pause(1)"""

        # Values at east, west, north, south quadrature points
        quad_E = state_E + high_ord_E
        quad_W = state_W + high_ord_W

        quad_N = state_N + high_ord_N
        quad_S = state_S + high_ord_S

        # ------------------------------------------------------------------------------
        # Venkatakrishnan Limiter here for now

        # u_max for interior cells
        u_max = np.maximum(refBLK.state.U,
                           np.maximum(np.maximum(state_E[:, :-1, :], state_W[:, 1:, :]),
                                      np.maximum(state_N[:-1, :, :], state_S[1:, :, :]))
                           )

        # u_min for interior cells
        u_min = np.minimum(refBLK.state.U,
                           np.minimum(np.minimum(state_E[:, :-1, :], state_W[:, 1:, :]),
                                      np.minimum(state_N[:-1, :, :], state_S[1:, :, :]))
                           )

        # Difference between quadrature point and cell average
        diff_E = quad_E[:, 1:, :]  - refBLK.state.U
        diff_W = quad_W[:, :-1, :] - refBLK.state.U
        diff_N = quad_N[1:, :, :]  - refBLK.state.U
        diff_S = quad_S[:-1, :, :] - refBLK.state.U

        # Difference between min/max and cell average
        diff_max = u_max - refBLK.state.U
        diff_min = u_min - refBLK.state.U

        #plt.contourf(refBLK.mesh.x, refBLK.mesh.y, (diff_min)[:, :, 0], 50, cmap='magma')
        #plt.pause(0.01)

        # Indices for ui,q - ui
        phiE = np.ones_like(refBLK.state.U)
        phiW = np.ones_like(refBLK.state.U)
        phiN = np.ones_like(refBLK.state.U)
        phiS = np.ones_like(refBLK.state.U)

        # Get phi
        y_max = diff_max / (diff_E + 1e-8)
        y_min = diff_min / (diff_E + 1e-8)
        phiE = np.where(diff_E > 0, (y_max ** 2 + 2 * y_max) / (y_max ** 2 + y_max + 2), phiE)
        phiE = np.where(diff_E < 0, (y_min ** 2 + 2 * y_min) / (y_min ** 2 + y_min + 2), phiE)

        y_max = diff_max / (diff_W + 1e-8)
        y_min = diff_min / (diff_W + 1e-8)
        phiW = np.where(diff_W > 0, (y_max ** 2 + 2 * y_max) / (y_max ** 2 + y_max + 2), phiW)
        phiW = np.where(diff_W < 0, (y_min ** 2 + 2 * y_min) / (y_min ** 2 + y_min + 2), phiW)

        y_max = diff_max / (diff_N + 1e-8)
        y_min = diff_min / (diff_N + 1e-8)
        phiN = np.where(diff_N > 0, (y_max ** 2 + 2 * y_max) / (y_max ** 2 + y_max + 2), phiN)
        phiN = np.where(diff_N < 0, (y_min ** 2 + 2 * y_min) / (y_min ** 2 + y_min + 2), phiN)

        y_max = diff_max / (diff_S + 1e-8)
        y_min = diff_min / (diff_S + 1e-8)
        phiS = np.where(diff_S > 0, (y_max ** 2 + 2 * y_max) / (y_max ** 2 + y_max + 2), phiS)
        phiS = np.where(diff_S < 0, (y_min ** 2 + 2 * y_min) / (y_min ** 2 + y_min + 2), phiS)

        phi = np.minimum(np.minimum(phiE, phiW), np.minimum(phiN, phiS))
        phi = np.where(phi < 0, 0, phi)

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
