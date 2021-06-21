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
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from abc import abstractmethod

from pyHype.limiters import limiters
from pyHype.states.states import ConservativeState
from pyHype.flux.Roe import ROE_FLUX_X
from pyHype.flux.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y
import pyHype.fvm.Gradients as Grads
import pyHype.utils.utils as utils


__DEFINED_FLUX_FUNCTIONS__ = ['Roe', 'HLLE', 'HLLL']
__DEFINED_SLOPE_LIMITERS__ = ['VanAlbada', 'VanLeer', 'Venkatakrishnan', 'BarthJespersen']
__DEFINED_GRADIENT_FUNCS__ = ['GreenGauss']
__DEFINED_RECONSTRUCTION__ = ['Primitive', 'Conservative']


class MUSCLFiniteVolumeMethod:
    def __init__(self,
                 inputs,
                 global_nBLK: int
                 ) -> None:
        """
        Solves the euler equations using a MUSCL-type finite volume scheme.

        TODO:
        ------ DESCRIBE MUSCL BRIEFLY ------

        The matrix structure used for storing solution data in various State classes is a (ny * nx * 4) numpy ndarray
        which has planar dimentions equal to the number of cells in the y and x direction, and a depth of 4. The
        structure looks as follows:

            ___________________nx____________________
            v                                       v
        |>  O----------O----------O----------O----------O ........................ q0 (zeroth state variable)
        |   |          |          |          |          |\
        |   |          |          |          |          |-O ...................... q1 (first state variable)
        |   |          |          |          |          | |\
        |   O----------O----------O----------O----------O |-O .................... q2 (second state variable)
        |   |          |          |          |          |\| |\
        |   |          |          |          |          |-O |-O .................. q3 (third state variable)
        |   |          |          |          |          | |\| |
        ny  O----------O----------O----------O----------O |-O |
        |   |          |          |          |          |\| |\|
        |   |          |          |          |          |-O |-O
        |   |          |          |          |          | |\| |
        |   O----------O----------O----------O----------O |-O |
        |   |          |          |          |          |\| |\|
        |   |          |          |          |          |-O | O
        |   |          |          |          |          | |\| |
        |>  O----------O----------O----------O----------O |-O |
             \|         \|         \|         \|         \| |\|
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O


        then, cells are constructed as follows:

        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |         |         |         |
        |         |         |         |         |
        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |    .....x.....    |         | -- Y+1/2
        |         |    .    |    .    |         |
        O---------O----x--- C ---x----O---------O -- Y
        |         |    .    |    .    |         |
        |         |    .....x.....    |         | -- Y-1/2
        |         |         |         |         |
        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |         |         |         |
        |         |         |         |         |
        O---------O---------O---------O---------O
                       |    |    |
                   X-1/2    X    X+1/2

        Reduction to 1D problem for each cell:

        x - direction:

        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |         |         |         |
        |         |         |         |         |
        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |         |         |         |
        |         |         |         |         |
        O---------O---------O---------O---------O
        |         |         |         |         |
        |         |         |         |         |
      ..|.........|.........|.........|.........|..
      . O----x----O----x--- C ---x----O----x----0 .
      ..|.........|.........|.........|.........|..
        |         |         |         |         |
        |         |         |         |         |
        O---------O---------O---------O---------0

        y - direction:
                          . . .
        O---------O-------.-O-.-------O---------O
        |         |       . | .       |         |
        |         |       . x .       |         |
        |         |       . | .       |         |
        O---------O-------.-O-.-------O---------O
        |         |       . | .       |         |
        |         |       . x .       |         |
        |         |       . | .       |         |
        O---------O-------.-C-.-------O---------O
        |         |       . | .       |         |
        |         |       . x .       |         |
        |         |       . | .       |         |
        O---------O-------.-O-.-------O---------O
        |         |       . | .       |         |
        |         |       . x .       |         |
        |         |       . | .       |         |
        O---------O-------.-O-.-------O---------O
                          . . .
        """

        # Set x and y direction number of points
        self.nx = inputs.nx
        self.ny = inputs.ny

        # Set inputs
        self.inputs = inputs

        # Set global block number
        self.global_nBLK = global_nBLK

        # Initialize x and y direction flux
        self.Flux_EW = np.empty((self.ny, self.nx + 1, 4))
        self.Flux_NS = np.empty((self.ny + 1, self.nx, 4))

        # Initialize left and right conservative states
        self.UL = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)
        self.UR = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)

        # Set Flux Function. Flux Function must be included in __DEFINED_FLUX_FUNCTIONS__
        _flux_func = self.inputs.flux_function

        if _flux_func in __DEFINED_FLUX_FUNCTIONS__:

            # ROE Flux
            if _flux_func == 'Roe':
                self.flux_function_X = ROE_FLUX_X(self.inputs, self.inputs.nx)
                self.flux_function_Y = ROE_FLUX_X(self.inputs, self.inputs.ny)
            # HLLE Flux
            elif _flux_func == 'HLLE':
                self.flux_function_X = HLLE_FLUX_X(self.inputs)
                self.flux_function_Y = HLLE_FLUX_Y(self.inputs)
            # HLLL Flux
            elif _flux_func == 'HLLL':
                self.flux_function_X = HLLL_FLUX_X(self.inputs)
                self.flux_function_Y = HLLL_FLUX_Y(self.inputs)
        # None
        else:
            raise ValueError('MUSCLFiniteVolumeMethod: Flux function type not specified.')

        # Set slope limiter. Slope limiter must be included in __DEFINED_SLOPE_LIMITERS__
        _flux_limiter = self.inputs.limiter

        if _flux_limiter in __DEFINED_SLOPE_LIMITERS__:

            # Van Leer limiter
            if _flux_limiter == 'VanLeer':
                self.flux_limiter = limiters.VanLeer(self.inputs)
            # Van Albada limiter
            elif _flux_limiter == 'VanAlbada':
                self.flux_limiter = limiters.VanAlbada(self.inputs)
            # Venkatakrishnan
            elif _flux_limiter == 'Venkatakrishnan':
                self.flux_limiter = limiters.Venkatakrishnan(self.inputs)
            # BarthJespersen
            elif _flux_limiter == 'BarthJespersen':
                self.flux_limiter = limiters.BarthJespersen(self.inputs)
        # None
        else:
            raise ValueError('MUSCLFiniteVolumeMethod: Slope limiter type not specified.')

        # Set slope limiter. Slope limiter must be included in __DEFINED_SLOPE_LIMITERS__
        _gradient = self.inputs.gradient

        if _gradient in __DEFINED_GRADIENT_FUNCS__:

            # Van Leer limiter
            if _gradient == 'GreenGauss':
                self.gradient = Grads.GreenGauss(self.inputs)
            # None
            else:
                raise ValueError('MUSCLFiniteVolumeMethod: Slope limiter type not specified.')

    # ------------------------------------------------------------------------------------------------------------------
    # Reconstruction functions

    def reconstruct(self,
                    refBLK
                    ) -> [np.ndarray]:
        """
        This method routes the state required for reconstruction to the correct implementation of the reconstruction.
        Current reconstruction methods are Primitive and Conservative.

        Parameters:
            - refBLK: Reference block to reconstruct

        Return:
            - stateE: Reconstructed state on east cell face
            - stateW: Reconstructed state on west cell face
            - stateN: Reconstructed state on north cell face
            - stateS: Reconstructed state on south cell face
        """

        # Select correct reconstruction type and return left and right reconstructed conservative states

        # Primitive reconstruction
        if self.inputs.reconstruction_type == 'Primitive':
            return self.reconstruct_primitive(refBLK)

        # Conservative reconstruction (by default)
        else:
            return self.reconstruct_state(refBLK, refBLK.state.U,
                                          refBLK.ghost.E.state.U, refBLK.ghost.W.state.U,
                                          refBLK.ghost.N.state.U, refBLK.ghost.S.state.U)


    def reconstruct_primitive(self,
                              refBLK,
                              ) -> [np.ndarray]:
        """
        Primitive reconstruction implementation. Simply convert the input ConservativeState into PrimitiveState and
        call the reconstruct_state implementation.

        Parameters:
            - U: Input ConservativeState for reconstruction.

        Return:
            - stateL: Left reconstructed conservative state
            - stateR: Right reconstructed conservative state
        """

        _state          = refBLK.state.to_primitive_vector()
        _state_E_ghost  = refBLK.ghost.E.state.to_primitive_vector()
        _state_W_ghost  = refBLK.ghost.W.state.to_primitive_vector()
        _state_N_ghost  = refBLK.ghost.N.state.to_primitive_vector()
        _state_S_ghost  = refBLK.ghost.S.state.to_primitive_vector()

        stateE, stateW, stateN, stateS = self.reconstruct_state(refBLK, _state,
                                                                _state_E_ghost, _state_W_ghost,
                                                                _state_N_ghost, _state_S_ghost)

        _state_E = ConservativeState(inputs=self.inputs, W_vector=stateE).U
        _state_W = ConservativeState(inputs=self.inputs, W_vector=stateW).U
        _state_N = ConservativeState(inputs=self.inputs, W_vector=stateN).U
        _state_S = ConservativeState(inputs=self.inputs, W_vector=stateS).U

        return _state_E, _state_W, _state_N, _state_S


    @abstractmethod
    def reconstruct_state(self,
                          refBLK,
                          state: np.ndarray,
                          ghostE: np.ndarray,
                          ghostW: np.ndarray,
                          ghostN: np.ndarray,
                          ghostS: np.ndarray
                          ) -> [np.ndarray]:
        """
        Implementation of the reconstruction method specialized to the Finite Volume Method described in the class.
        """
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # Flux evaluation and integration functions

    @abstractmethod
    def integrate_flux_E(self, refBLK):
        pass


    @abstractmethod
    def integrate_flux_W(self, refBLK):
        pass


    @abstractmethod
    def integrate_flux_N(self, refBLK):
        pass


    @abstractmethod
    def integrate_flux_S(self, refBLK):
        pass


    def get_residual(self, refBLK):
        """
        Compute residuals used for marching the solution through time by integrating the fluxes on each cell face and
        applying the semi-discrete Godunov method:

        dUdt[i] = - (1/A[i]) * sum[over all faces] (F[face] * length[face])
        """

        # Compute fluxes
        self.get_flux(refBLK)

        # Integrate fluxes
        fluxE = self.integrate_flux_E(refBLK)
        fluxW = self.integrate_flux_W(refBLK)
        fluxN = self.integrate_flux_N(refBLK)
        fluxS = self.integrate_flux_S(refBLK)

        return -(fluxE + fluxW + fluxN + fluxS) / refBLK.mesh.A


    def get_flux(self, refBLK):
        """
        Compute the flux at each cell face by sweeping through rows and columns of the domain.
        """

        # Compute x and y direction gradients
        self.gradient(refBLK)

        # Get reconstructed quadrature points
        stateE, stateW, stateN, stateS = self.reconstruct(refBLK)

        # --------------------------------------------------------------------------------------------------------------
        # Calculate x-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in x-direction
        self.UL.reset(shape=(1, self.nx + 1, 4))
        self.UR.reset(shape=(1, self.nx + 1, 4))

        # Rotate to allign with cell faces
        utils.rotate(refBLK.mesh.thetax, stateE, stateW)

        # Iterate over all rows in block
        for row in range(self.ny):

            # Set vectors based on left and right states
            self.UL.from_conservative_state_vector(stateE[row:row+1, :, :])
            self.UR.from_conservative_state_vector(stateW[row:row+1, :, :])

            # Calculate face-normal-flux at each cell interface
            self.Flux_EW[row, :, :] = self.flux_function_X.compute_flux(self.UL, self.UR)

        # Rotate flux back to local frame
        utils.unrotate(refBLK.mesh.thetax, self.Flux_EW)

        # --------------------------------------------------------------------------------------------------------------
        # Calculate y-direction Flux

        # Reset U vector holder sizes to ensure compatible with number of cells in y-direction
        self.UL.reset(shape=(1, self.ny + 1, 4))
        self.UR.reset(shape=(1, self.ny + 1, 4))

        # Rotate to allign with cell faces
        utils.rotate(refBLK.mesh.thetay[:, np.newaxis], stateN, stateS)

        # Transpose North and South states
        stateN = stateN.transpose((1, 0, 2))
        stateS = stateS.transpose((1, 0, 2))

        # Iterate over all columns in block
        for col in range(self.nx):

            # Set vectors based on left and right states
            self.UL.from_conservative_state_vector(stateN[col:col + 1, :, :])
            self.UR.from_conservative_state_vector(stateS[col:col + 1, :, :])

            # Calculate face-normal-flux at each cell interface
            self.Flux_NS[:, col, :] = self.flux_function_Y.compute_flux(self.UL, self.UR).reshape(-1, 4)

        # Rotate flux back to global frame
        utils.unrotate(refBLK.mesh.thetay[:, np.newaxis], self.Flux_NS)
