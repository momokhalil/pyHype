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
from __future__ import annotations

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import numba as nb
from abc import abstractmethod

from pyHype.limiters import limiters
from pyHype.states.states import State, ConservativeState
from pyHype.flux.Roe import ROE_FLUX_X
from pyHype.flux.HLLE import HLLE_FLUX_X
from pyHype.flux.HLLL import HLLL_FLUX_X
import pyHype.fvm.Gradients as Grads
import pyHype.utils.utils as utils
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock
    from pyHype.blocks.base import GhostBlockContainer
    from pyHype.mesh.base import Mesh


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
        self.Flux_E = np.empty((self.ny, self.nx, 4))
        self.Flux_W = np.empty((self.ny, self.nx, 4))
        self.Flux_N = np.empty((self.ny, self.nx, 4))
        self.Flux_S = np.empty((self.ny, self.nx, 4))

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
                self.flux_function_Y = HLLE_FLUX_X(self.inputs)
            # HLLL Flux
            elif _flux_func == 'HLLL':
                self.flux_function_X = HLLL_FLUX_X(self.inputs)
                self.flux_function_Y = HLLL_FLUX_X(self.inputs)
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
        if self.inputs.reconstruction_type == 'primitive':
            _to_recon = refBLK.to_primitive(copy=True)
        else:
            _to_recon = refBLK

        # Get gradients
        self.gradient(_to_recon)

        # Reconstruct state
        _stateE, _stateW, _stateN, _stateS = self.reconstruct_state(_to_recon)

        # Conservative reconstruction (by default)
        return _stateE, _stateW, _stateN, _stateS, _to_recon


    @abstractmethod
    def reconstruct_state(self,
                          refBLK: QuadBlock
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


    def dUdt(self, refBLK):
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


    def get_flux(self, refBLK: QuadBlock):

        # Get reconstructed quadrature points
        _stateE, _stateW, _stateN, _stateS, _to_recon = self.reconstruct(refBLK)

        # Calculate x-direction Flux

        # Copy all ghost cell values that will be used for the flux calculations
        _ghostE = _to_recon.ghost.E.col(0, copy=True)
        _ghostW = _to_recon.ghost.W.col(-1, copy=True)
        _ghostN = _to_recon.ghost.N.row(0, copy=True)
        _ghostS = _to_recon.ghost.S.row(-1, copy=True)

        # Rotate to allign with cell faces
        if not _to_recon.is_cartesian:
            utils.rotate(_to_recon.mesh.faceE.theta, _stateE)
            utils.rotate(_to_recon.mesh.faceW.theta - np.pi, _stateW)
            utils.rotate(_to_recon.mesh.get_east_face_angle(), _ghostE)
            utils.rotate(_to_recon.mesh.get_west_face_angle(), _ghostW)

        # Get states on the left and right on EW interfaces
        _stateL = np.concatenate((_ghostW, _stateE), axis=1)
        _stateR = np.concatenate((_stateW, _ghostE), axis=1)

        # Iterate over all rows in block
        for row in range(self.ny):
            # Get cell interface flux
            if refBLK.reconstruction_type == 'primitive':
                flux_EW = self.flux_function_X(WL=_stateL[row, None, :, :], 
                                               WR=_stateR[row, None, :, :])
            else:
                flux_EW = self.flux_function_X(UL=_stateL[row, None, :, :],
                                               UR=_stateR[row, None, :, :])
            # Set east face flux
            self.Flux_E[row, :, :] = flux_EW[:, 1:, :]
            # Set west face flux
            self.Flux_W[row, :, :] = flux_EW[:, :-1, :]

        # Rotate flux back to local frame
        if not _to_recon.is_cartesian:
            utils.unrotate(_to_recon.mesh.faceE.theta, self.Flux_E)
            utils.unrotate(_to_recon.mesh.faceW.theta - np.pi, self.Flux_W)

        # Calculate y-direction Flux
        f = 0.00
        # Rotate to allign with cell faces
        if _to_recon.is_cartesian:
            # If block is cartesian, rotate by 90 degrees CCW (implemented as array swaps for efficiency)
            utils.rotate90(_stateN, _stateS, _ghostN, _ghostS)
        else:
            # If not, rotate by the given angle using the standard rotation matrix
            utils.rotate(_to_recon.mesh.faceN.theta, _stateN)
            utils.rotate(_to_recon.mesh.faceS.theta - np.pi, _stateS)
            utils.rotate(_to_recon.mesh.get_north_face_angle(), _ghostN)
            utils.rotate(_to_recon.mesh.get_south_face_angle(), _ghostS)

        # Get states on the left and right on NS interfaces
        _stateL = np.concatenate((_ghostS, _stateN), axis=0).transpose((1, 0, 2))
        _stateR = np.concatenate((_stateS, _ghostN), axis=0).transpose((1, 0, 2))

        # Iterate over all columns in block
        for col in range(self.nx):
            # Calculate face-normal-flux at each cell east-west interface
            if refBLK.reconstruction_type == 'primitive':
                flux_NS = self.flux_function_Y(WL=_stateL[col, None, :, :],
                                               WR=_stateR[col, None, :, :]
                                               ).reshape(-1, 4)
            else:
                flux_NS = self.flux_function_Y(UL=_stateL[col, None, :, :],
                                               UR=_stateR[col, None, :, :]
                                               ).reshape(-1, 4)
            # Set east face flux
            self.Flux_N[:, col, :] = flux_NS[1:, :]
            # Set west face flux
            self.Flux_S[:, col, :] = flux_NS[:-1, :]

        # Rotate flux back to global frame
        if refBLK.is_cartesian:
            # If block is cartesian, rotate by 90 degrees CW (implemented as array swaps for efficiency)
            utils.unrotate90(self.Flux_N, self.Flux_S)
        else:
            # If not, rotate by the given angle using the standard rotation matrix
            utils.unrotate(_to_recon.mesh.faceN.theta, self.Flux_N)
            utils.unrotate(_to_recon.mesh.faceS.theta - np.pi, self.Flux_S)

