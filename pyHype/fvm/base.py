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
from abc import abstractmethod

from pyHype.limiters import limiters
from pyHype.states.states import ConservativeState, State
from pyHype.flux.Roe import ROE_FLUX_X, ROE_FLUX_Y
from pyHype.flux.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y


__DEFINED_FLUX_FUNCTIONS__ = ['Roe', 'HLLE', 'HLLL']
__DEFINED_SLOPE_LIMITERS__ = ['VanAlbada', 'VanLeer', 'Venkatakrishnan', 'BarthJespersen']

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
        self.Flux_X = np.empty((self.ny, self.nx, 4))
        self.Flux_Y = np.empty((self.ny, self.nx, 4))

        # Initialize left and right conservative states
        self.UL = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)
        self.UR = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)

        # Set Flux Function. Flux Function must be included in __DEFINED_FLUX_FUNCTIONS__
        _flux_func = self.inputs.flux_function

        if _flux_func in __DEFINED_FLUX_FUNCTIONS__:

            # ROE Flux
            if _flux_func == 'Roe':
                self.flux_function_X = ROE_FLUX_X(self.inputs)
                self.flux_function_Y = ROE_FLUX_Y(self.inputs)
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
        _flux_limiter = self.inputs.flux_limiter

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

    def reconstruct(self,
                    U: ConservativeState
                    ) -> None:
        """
        This method routes the state required for reconstruction to the correct implementation of the reconstruction.
        Current reconstruction methods are Primitive and Conservative.

        Parameters:
            - U: Input ConservativeState that needs reconstruction.

        Return:
            N.A
        """

        # Select correct reconstruction type and return left and right reconstructed conservative states

        # Primitive reconstruction
        if self.inputs.reconstruction_type == 'Primitive':
            stateL, stateR = self.reconstruct_primitive(U)
            self.UL.from_primitive_state_vector(stateL)
            self.UR.from_primitive_state_vector(stateR)

        # Conservative reconstruction
        elif self.inputs.reconstruction_type == 'Conservative':
            stateL, stateR = self.reconstruct_conservative(U)
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)

        # Default to Conservative
        else:
            stateL, stateR = self.reconstruct_conservative(U)
            self.UL.from_conservative_state_vector(stateL)
            self.UR.from_conservative_state_vector(stateR)


    def reconstruct_primitive(self,
                              U: ConservativeState
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

        _to_construct = U.to_primitive_state()
        stateL, stateR = self.reconstruct_state(_to_construct)

        return stateL, stateR

    def reconstruct_conservative(self,
                                 U: ConservativeState
                                 ) -> [np.ndarray]:
        """
        Conservative reconstruction implementation. Simply pass the input ConservativeState into the
        reconstruct_state implementation.

        Parameters:
            - U: Input ConservativeState for reconstruction.

        Return:
            - stateL: Left reconstructed conservative state
            - stateR: Right reconstructed conservative state
        """

        stateL, stateR = self.reconstruct_state(U)

        return stateL, stateR

    def get_interface_values(self, refBLK):

        if self.inputs.interface_interpolation == 'arithmetic_average':
            interfaceEW, interfaceNS = self.get_interface_values_arithmetic(refBLK)
            return interfaceEW, interfaceNS
        else:
            raise ValueError('Interface Interpolation method is not defined.')

    def get_interface_values_arithmetic(self, refBLK) -> [np.ndarray]:

        # Concatenate mesh state and ghost block states
        if self.inputs.reconstruction_type == 'Primitive':
            _W = refBLK.state.get_W_array()
            catx = np.concatenate((refBLK.ghost.W.state.get_W_array(),
                                   _W,
                                   refBLK.ghost.E.state.get_W_array()),
                                  axis=1)

            caty = np.concatenate((refBLK.ghost.N.state.get_W_array(),
                                   _W,
                                   refBLK.ghost.S.state.get_W_array()),
                                  axis=0)

        elif self.inputs.reconstruction_type == 'Conservative':
            catx = np.concatenate((refBLK.ghost.W.state.U,
                                   refBLK.state.U,
                                   refBLK.ghost.E.state.U),
                                  axis=1)

            caty = np.concatenate((refBLK.ghost.S.state.U,
                                   refBLK.state.U,
                                   refBLK.ghost.N.state.U),
                                  axis=0)

        else:
            raise ValueError('Undefined reconstruction type')

        # Compute arithmetic mean
        interfaceEW = 0.5 * (catx[:, 1:, :] + catx[:, :-1, :])
        interfaceNS = 0.5 * (caty[1:, :, :] + caty[:-1, :, :])

        return interfaceEW, interfaceNS

    @abstractmethod
    def reconstruct_state(self,
                          U: State
                          ) -> [np.ndarray]:
        """
        Implementation of the reconstruction method specialized to the Finite Volume Method described in the class.
        """
        pass

    @abstractmethod
    def get_grad(self, refBLK):
        pass
