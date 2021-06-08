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

        _ZERO_COL = np.zeros((refBLK.mesh.ny, 1, 4))

        #stateL = np.concatenate((refBLK.ghost.W.state.U, refBLK.state.U), axis=1)

        high_ord_L = gradx * (refBLK.mesh.east_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.east_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        #stateR = np.concatenate((refBLK.state.U, refBLK.ghost.E.state.U), axis=1)

        high_ord_R = gradx * (refBLK.mesh.west_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.west_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        return np.concatenate((_ZERO_COL, high_ord_L), axis=1), np.concatenate((high_ord_R, _ZERO_COL), axis=1)

    @staticmethod
    def high_order_term_NS(refBLK, gradx, grady):

        _ZERO_ROW = np.zeros((1, refBLK.mesh.nx, 4))

        high_ord_L = gradx * (refBLK.mesh.south_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.south_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        high_ord_R = gradx * (refBLK.mesh.north_face_midpoint_x - refBLK.mesh.x[:, :, np.newaxis]) \
                   + grady * (refBLK.mesh.north_face_midpoint_y - refBLK.mesh.y[:, :, np.newaxis])

        return np.concatenate((high_ord_L, _ZERO_ROW), axis=0), np.concatenate((_ZERO_ROW, high_ord_R), axis=0)


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

        # Values at east, west, north, south quadrature points
        quad_W = state_EW_R + high_ord_EW_R
        quad_E = state_EW_L + high_ord_EW_L

        quad_N = state_NS_L + high_ord_NS_L
        quad_S = state_NS_R + high_ord_NS_R

        # ------------------------------------------------------------------------------
        # Barth-Jespersen Limiter here for now

        # u_max for interior cells
        max_avg = np.maximum(np.maximum(state_EW_L[:, :-1, :], state_EW_R[:, 1:, :]),
                             np.maximum(state_NS_L[:-1, :, :], state_NS_R[1:, :, :]))

        u_max = np.maximum(refBLK.state.U, max_avg)

        # u_min for interior cells
        min_avg = np.minimum(np.minimum(state_EW_L[:, :-1, :], state_EW_R[:, 1:, :]),
                             np.minimum(state_NS_L[:-1, :, :], state_NS_R[1:, :, :]))

        u_min = np.minimum(refBLK.state.U, min_avg)

        # u_max for boundaries
        max_avg_E = np.maximum(refBLK.state.U[:, -1:, :], refBLK.ghost.E.state.U)
        max_avg_W = np.maximum(refBLK.state.U[:, :0, :], refBLK.ghost.W.state.U)
        max_avg_S = np.maximum(refBLK.state.U[:0, :, :], refBLK.ghost.S.state.U)
        max_avg_N = np.maximum(refBLK.state.U[-1:, :, :], refBLK.ghost.N.state.U)

        # Difference between quadrature point and cell average
        diff_E = quad_E[:, 1:, :]  - refBLK.state.U
        diff_W = quad_W[:, :-1, :] - refBLK.state.U
        diff_N = quad_N[1:, :, :]  - refBLK.state.U
        diff_S = quad_S[:-1, :, :] - refBLK.state.U

        # Difference between min/max and cell average
        diff_max = u_max - refBLK.state.U
        diff_min = u_min - refBLK.state.U

        # Indices for ui,q - ui
        phiE = np.zeros_like(refBLK.state.U)
        phiW = np.zeros_like(refBLK.state.U)
        phiN = np.zeros_like(refBLK.state.U)
        phiS = np.zeros_like(refBLK.state.U)

        """phiE = np.where(diff_E > 0, np.minimum(1, diff_max / (diff_E + 1e-8)), phiE)
        phiE = np.where(diff_E < 0, np.minimum(1, diff_min / (diff_E + 1e-8)), phiE)
        phiE = np.where(diff_E == 0, 1, phiE)

        phiW = np.where(diff_W > 0, np.minimum(1, diff_max / (diff_W + 1e-8)), phiW)
        phiW = np.where(diff_W < 0, np.minimum(1, diff_min / (diff_W + 1e-8)), phiW)
        phiW = np.where(diff_W == 0, 1, phiW)

        phiN = np.where(diff_N > 0, np.minimum(1, diff_max / (diff_N + 1e-8)), phiN)
        phiN = np.where(diff_N < 0, np.minimum(1, diff_min / (diff_N + 1e-8)), phiN)
        phiN = np.where(diff_N == 0, 1, phiN)

        phiS = np.where(diff_S > 0, np.minimum(1, diff_max / (diff_S + 1e-8)), phiS)
        phiS = np.where(diff_S < 0, np.minimum(1, diff_min / (diff_S + 1e-8)), phiS)
        phiS = np.where(diff_S == 0, 1, phiS)
        
        phi = np.minimum(np.minimum(phiE, phiW), np.minimum(phiN, phiS))"""

        # Venkatakrishnan
        y_max = diff_max / (diff_E + 1e-8)
        y_min = diff_min / (diff_E + 1e-8)
        phiE = np.where(diff_E > 0, (y_max**2 + 2*y_max)/(y_max**2 + y_max + 2), phiE)
        phiE = np.where(diff_E < 0, (y_min**2 + 2*y_min)/(y_min**2 + y_min + 2), phiE)
        phiE = np.where(diff_E == 0, 1, phiE)

        y_max = diff_max / (diff_W + 1e-8)
        y_min = diff_min / (diff_W + 1e-8)
        phiW = np.where(diff_W > 0, (y_max**2 + 2*y_max)/(y_max**2 + y_max + 2), phiW)
        phiW = np.where(diff_W < 0, (y_min**2 + 2*y_min)/(y_min**2 + y_min + 2), phiW)
        phiW = np.where(diff_W == 0, 1, phiW)

        y_max = diff_max / (diff_N + 1e-8)
        y_min = diff_min / (diff_N + 1e-8)
        phiN = np.where(diff_N > 0, (y_max**2 + 2*y_max)/(y_max**2 + y_max + 2), phiN)
        phiN = np.where(diff_N < 0, (y_min**2 + 2*y_min)/(y_min**2 + y_min + 2), phiN)
        phiN = np.where(diff_N == 0, 1, phiN)

        y_max = diff_max / (diff_S + 1e-8)
        y_min = diff_min / (diff_S + 1e-8)
        phiS = np.where(diff_S > 0, (y_max**2 + 2*y_max)/(y_max**2 + y_max + 2), phiS)
        phiS = np.where(diff_S < 0, (y_min**2 + 2*y_min)/(y_min**2 + y_min + 2), phiS)
        phiS = np.where(diff_S == 0, 1, phiS)

        phi = np.minimum(np.minimum(phiE, phiW), np.minimum(phiN, phiS))

        print(phi[:, :, 1])



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

        """print(refBLK.ghost.N.state.U[:, :, 0].shape, refBLK.ghost.N.state.rho.shape)
        print(refBLK.ghost.S.state.U[:, :, 0].shape)
        print(refBLK.ghost.E.state.U[:, :, 0].shape)
        print(refBLK.ghost.W.state.U[:, :, 0].shape)

        print(refBLK.state.U[:, :, 0].shape, refBLK.state.rho.shape)
"""
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
