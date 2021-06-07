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

    @staticmethod
    def high_order_term_EW(refBLK, gradx, grady):

        #stateL = np.concatenate((refBLK.ghost.W.state.U, refBLK.state.U), axis=1)

        high_ord_L = gradx * (refBLK.mesh.east_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.east_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        #stateR = np.concatenate((refBLK.state.U, refBLK.ghost.E.state.U), axis=1)

        high_ord_R = gradx * (refBLK.mesh.west_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.west_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        return high_ord_L, high_ord_R

    @staticmethod
    def high_order_term_NS(refBLK, gradx, grady):

        high_ord_L = gradx * (refBLK.mesh.south_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.south_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        high_ord_R = gradx * (refBLK.mesh.north_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.north_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        return high_ord_L, high_ord_R


    def get_flux(self, refBLK):
        """
        Compute the flux at each cell center using the Green Gauss reconstruction method and the approximate Riemann
        solver and slope limiter of choice.
        """

        # Compute x and y direction gradients
        gradx, grady = self.get_grad(refBLK)

        # East-West high order term for left and right states on each east-west cell interface
        high_ord_EW_L, high_ord_EW_R = self.high_order_term_EW(refBLK, gradx, grady)

        # North-South high order term for left and right states on each north-south cell interface
        high_ord_NS_L, high_ord_NS_R = self.high_order_term_NS(refBLK, gradx, grady)

        # East-West direction left and right states
        state_EW_L = np.concatenate((refBLK.ghost.W.state.U, refBLK.state.U), axis=1)
        state_EW_R = np.concatenate((refBLK.state.U, refBLK.ghost.E.state.U), axis=1)

        # North-South direction left and right states
        state_NS_L = np.concatenate((refBLK.state.U, refBLK.ghost.S.state.U), axis=0)
        state_NS_R = np.concatenate((refBLK.ghost.N.state.U, refBLK.state.U), axis=0)


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


    def reconstruct_state(self, state: np.ndarray) -> [np.ndarray]:


        return stateL, stateR

    def get_grad(self, refBLK):

        print(refBLK.ghost.N.state.U[:, :, 0].shape, refBLK.ghost.N.state.rho.shape)
        print(refBLK.ghost.S.state.U[:, :, 0].shape)
        print(refBLK.ghost.E.state.U[:, :, 0].shape)
        print(refBLK.ghost.W.state.U[:, :, 0].shape)

        print(refBLK.state.U[:, :, 0].shape, refBLK.state.rho.shape)

        # Concatenate mesh state and ghost block states
        interfaceE, interfaceW, interfaceN, interfaceS = self.get_interface_values(refBLK)

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
