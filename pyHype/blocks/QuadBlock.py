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
from typing import TYPE_CHECKING

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pyHype.fvm import FirstOrder, SecondOrderPWL
from pyHype.mesh.base import BlockDescription, Mesh
from pyHype.states.states import ConservativeState, PrimitiveState
from pyHype.blocks.base import NormalVector, GhostBlockContainer, Neighbors
from pyHype.solvers.time_integration.explicit_runge_kutta import ExplicitRungeKutta as Erk
from pyHype.blocks.ghost import GhostBlockEast, GhostBlockWest, GhostBlockSouth, GhostBlockNorth

from copy import deepcopy
from copy import copy as cpy

from itertools import chain


if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput

# QuadBlock Class Definition
class QuadBlock:
    def __init__(self,
                 inputs: ProblemInput,
                 block_data: BlockDescription
                 ) -> None:

        self.inputs             = inputs
        self.block_data         = block_data
        self.mesh               = Mesh(inputs, block_data)
        self.state              = ConservativeState(inputs, nx=inputs.nx, ny=inputs.ny)
        self.global_nBLK        = block_data.nBLK
        self.ghost              = None
        self.neighbors          = None
        self.nx                 = inputs.nx
        self.ny                 = inputs.ny


        # Store vertices for brevity
        vert = self.mesh.vertices

        # Side lengths
        self.LE                 = self._get_side_length(vert.SE, vert.NE)
        self.LW                 = self._get_side_length(vert.SW, vert.NW)
        self.LS                 = self._get_side_length(vert.SE, vert.SW)
        self.LS                 = self._get_side_length(vert.NW, vert.NE)

        # Side angles
        self.thetaE             = self._get_side_angle(vert.SE, vert.NE)
        self.thetaW             = self._get_side_angle(vert.SW, vert.NW) + np.pi
        self.thetaS             = self._get_side_angle(vert.SW, vert.SE) + np.pi
        self.thetaN             = self._get_side_angle(vert.NE, vert.NW)

        # Self normal vectors
        self.nE                 = NormalVector(self.thetaE)
        self.nW                 = NormalVector(self.thetaW)
        self.nS                 = NormalVector(self.thetaS)
        self.nN                 = NormalVector(self.thetaN)

        # Conservative Gradient
        if self.inputs.reconstruction_type == 'conservative':
            self.dUdx               = np.zeros_like(self.mesh.x, dtype=float)
            self.dUdy               = np.zeros_like(self.mesh.y, dtype=float)
            self.dWdx               = None
            self.dWdy               = None
        # Primitive Gradient
        elif self.inputs.reconstruction_type == 'primitive':
            self.dWdx               = np.zeros_like(self.mesh.x, dtype=float)
            self.dWdy               = np.zeros_like(self.mesh.y, dtype=float)
            self.dUdx               = None
            self.dUdy               = None
        else:
            raise ValueError('Reconstruction type is not defined.')

        # Set finite volume method
        fvm = self.inputs.fvm

        if fvm == 'FirstOrder':
            self.fvm = FirstOrder(self.inputs, self.global_nBLK)
        elif fvm == 'SecondOrderPWL':
            self.fvm = SecondOrderPWL(self.inputs, self.global_nBLK)
        else:
            raise ValueError('Specified finite volume method has not been specialized.')

        # Set time integrator
        time_integrator = self.inputs.integrator

        if time_integrator      == 'ExplicitEuler1':
            self._time_integrator = Erk.ExplicitEuler1(self.inputs)
        elif time_integrator    == 'RK2':
            self._time_integrator = Erk.RK2(self.inputs)
        elif time_integrator    == 'Generic2':
            self._time_integrator = Erk.Generic2(self.inputs)
        elif time_integrator    == 'Ralston2':
            self._time_integrator = Erk.Ralston2(self.inputs)
        elif time_integrator    == 'Generic3':
            self._time_integrator = Erk.Generic3(self.inputs)
        elif time_integrator    == 'RK3':
            self._time_integrator = Erk.RK3(self.inputs)
        elif time_integrator    == 'RK3SSP':
            self._time_integrator = Erk.RK3SSP(self.inputs)
        elif time_integrator    == 'Ralston3':
            self._time_integrator = Erk.Ralston3(self.inputs)
        elif time_integrator    == 'RK4':
            self._time_integrator = Erk.RK4(self.inputs)
        elif time_integrator    == 'Ralston4':
            self._time_integrator = Erk.Ralston4(self.inputs)
        elif time_integrator    == 'DormandPrince5':
            self._time_integrator = Erk.DormandPrince5(self.inputs)
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # Build boundary blocks
        self.ghost = GhostBlockContainer(E=GhostBlockEast(self.inputs, BCtype=block_data.BCTypeE, refBLK=self),
                                         W=GhostBlockWest(self.inputs, BCtype=block_data.BCTypeW, refBLK=self),
                                         N=GhostBlockNorth(self.inputs, BCtype=block_data.BCTypeN, refBLK=self),
                                         S=GhostBlockSouth(self.inputs, BCtype=block_data.BCTypeS, refBLK=self))

        # is block cartesian
        self.is_cartesian = self._is_cartesian()


    def _is_cartesian(self) -> bool:
        """
        Return boolen value that indicates if the block is alligned with the cartesian axes.

        Parameters:
            - None

        Return:
            - _is_cartesian (bool): Boolean that is True if the block is cartesian and False if it isnt
        """

        _is_cartesian = (self.mesh.vertices.NE[1] == self.mesh.vertices.NW[1]) and \
                        (self.mesh.vertices.SE[1] == self.mesh.vertices.SW[1]) and \
                        (self.mesh.vertices.SE[0] == self.mesh.vertices.NE[0]) and \
                        (self.mesh.vertices.SW[0] == self.mesh.vertices.NW[0])

        return _is_cartesian

    @property
    def reconstruction_type(self):
        """
        Returns the reconstruction type used in the finite volume method.

        Parameters:
            - None

        Return:
            - (str): the reconstruction type
        """

        return self.inputs.reconstruction_type

    def scopy(self):
        _cpy = cpy(self)
        _cpy.ghost = cpy(self.ghost)
        _cpy.ghost.E = cpy(self.ghost.E)
        _cpy.ghost.W = cpy(self.ghost.W)
        _cpy.ghost.N = cpy(self.ghost.N)
        _cpy.ghost.S = cpy(self.ghost.S)
        return _cpy

    def dcopy(self):
        return deepcopy(self)

    def to_primitive(self, copy: bool = True) -> QuadBlock:
        _to_conv = self.scopy() if copy else self
        self._to_primitive(_to_conv)
        return _to_conv

    def to_conservative(self, copy: bool = True) -> QuadBlock:
        _to_conv = self.scopy() if copy else self
        self._to_conservative(_to_conv)
        return _to_conv

    @staticmethod
    def _is_all_blk_conservative(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, ConservativeState), blks))

    @staticmethod
    def _is_all_blk_primitive(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, PrimitiveState), blks))

    def _to_primitive(self, block: QuadBlock):
        _ghost_dict_vals = block.ghost.__dict__.values()

        if not isinstance(block.state, ConservativeState):
            raise TypeError('Reference block state is not ConservativeState.')
        elif not self._is_all_blk_conservative(_ghost_dict_vals):
            raise TypeError('Ghost block state is not ConservativeState.')
        else:
            block.state = block.state.to_primitive_state()
            for v in _ghost_dict_vals:
                v.state = v.state.to_primitive_state()

    def _to_conservative(self, block: QuadBlock):
        _ghost_dict_vals = block.ghost.__dict__.values()

        if not isinstance(block.state, PrimitiveState):
            raise TypeError('Reference block state is not PrimitiveState.')
        elif not self._is_all_blk_primitive(_ghost_dict_vals):
            raise TypeError('Ghost block state is not PrimitiveState.')
        else:
            block.state = block.state.to_conservative_state()
            for v in _ghost_dict_vals:
                v.state = v.state.to_conservative_state()

    def plot(self,
             ax: plt.axes = None,
             show_cell_centre: bool = False):
        """
        # FOR DEBUGGING

        Plot mesh. Plots the nodes and cell center locations and connect them.

        Parameters:
            - None

        Returns:
            - None
        """

        _show = True if ax is None else False

        if not ax:
            fig, ax = plt.subplots(1, 1)

        # Create scatter plots for nodes
        ax.scatter(self.mesh.nodes.x[:, :, 0],
                    self.mesh.nodes.y[:, :, 0],
                    color='black',
                    s=0)

        # Create nodes mesh for LineCollection
        east = np.stack((self.mesh.nodes.x[:, -1, None, 0],
                          self.mesh.nodes.y[:, -1, None, 0]),
                         axis=2)

        west = np.stack((self.mesh.nodes.x[:, 0, None, 0],
                         self.mesh.nodes.y[:, 0, None, 0]),
                        axis=2)

        north = np.stack((self.mesh.nodes.x[-1, None, :, 0],
                         self.mesh.nodes.y[-1, None, :, 0]),
                         axis=2)

        south = np.stack((self.mesh.nodes.x[0, None, :, 0],
                         self.mesh.nodes.y[0, None, :, 0]),
                         axis=2)

        block_sides = chain((east, west, north, south))

        body = np.stack((self.mesh.nodes.x[:, :, 0],
                         self.mesh.nodes.y[:, :, 0]),
                        axis=2)

        # Create LineCollection for nodes
        ax.add_collection(LineCollection(body,
                                         colors='black',
                                         linewidths=1,
                                         alpha=1))
        ax.add_collection(LineCollection(body.transpose((1, 0, 2)),
                                         colors='black',
                                         linewidths=1,
                                         alpha=1))

        for side in block_sides:

            ax.add_collection(LineCollection(side,
                                             colors='black',
                                             linewidths=2,
                                             alpha=1))
            ax.add_collection(LineCollection(side.transpose((1, 0, 2)),
                                             colors='black',
                                             linewidths=2,
                                             alpha=1))


        if show_cell_centre:

            # Create scatter plots cell centers
            ax.scatter(self.mesh.x[:, :, 0],
                       self.mesh.y[:, :, 0],
                       color='mediumslateblue',
                       s=0,
                       alpha=0.5)

            # Create cell center mesh for LineCollection
            segs1 = np.stack((self.mesh.x[:, :, 0],
                              self.mesh.y[:, :, 0]),
                             axis=2)
            segs2 = segs1.transpose((1, 0, 2))

            # Create LineCollection for cell centers
            ax.add_collection(LineCollection(segs1,
                                                    colors='mediumslateblue',
                                                    linestyles='--',
                                                    linewidths=1,
                                                    alpha=0.5))
            ax.add_collection(LineCollection(segs2,
                                                    colors='mediumslateblue',
                                                    linestyles='--',
                                                    linewidths=1,
                                                    alpha=0.5))
        if _show:
            # Show Plot
            plt.show()

            # Close plot
            plt.close()

    @property
    def Flux_E(self):
        """
        Returns the flux arrays for the east face. Retrieves the arrays from the finite-volume-method class.

        Parameters:
            - None

        Return:
            - (np.ndarray): Numpy array containing the flux values for the east face.
        """

        return self.fvm.Flux_E

    @property
    def Flux_W(self):
        """
        Returns the flux arrays for the west face. Retrieves the arrays from the finite-volume-method class.

        Parameters:
            - None

        Return:
            - (np.ndarray): Numpy array containing the flux values for the west face.
        """
        return self.fvm.Flux_W

    @property
    def Flux_N(self):
        """
        Returns the flux arrays for the north face. Retrieves the arrays from the finite-volume-method class.

        Parameters:
            - None

        Return:
            - (np.ndarray): Numpy array containing the flux values for the north face.
        """

        return self.fvm.Flux_N

    @property
    def Flux_S(self):
        """
        Returns the flux arrays for the south face. Retrieves the arrays from the finite-volume-method class.

        Parameters:
            - None

        Return:
            - (np.ndarray): Numpy array containing the flux values for the south face.
        """

        return self.fvm.Flux_S

    @property
    def gradx(self):
        """
        Returns the x-direction gradients based on the reconstruction type. For example, if the reconstruction type is
        primitive, this will return self.dWdx.

        Parameters:
            - None

        Return:
            - (np.ndarray): Values of the x-direction gradients.
        """

        if self.reconstruction_type == 'primitive':
            return self.dWdx
        elif self.reconstruction_type == 'conservative':
            return self.dUdx
        else:
            raise ValueError('Reconstruction type ' + str(self.reconstruction_type) + ' is not defined.')

    @property
    def grady(self):
        """
        Returns the y-direction gradients based on the reconstruction type. For example, if the reconstruction type is
        primitive, this will return self.dWdy.

        Parameters:
            - None

        Return:
            - (np.ndarray): Values of the y-direction gradients.
        """
        if self.reconstruction_type == 'primitive':
            return self.dWdy
        elif self.reconstruction_type == 'conservative':
            return self.dUdy
        else:
            raise ValueError('Reconstruction type ' + str(self.reconstruction_type) + ' is not defined.')

    @gradx.setter
    def gradx(self, gradx):
        """
        Sets the x-direction gradients based on the reconstruction type. For example, if the reconstruction type is
        primitive, this will set the self.dWdx attribute.

        Parameters:
            - (np.ndarray): Values of the x-direction gradients.

        Return:
            - None
        """

        if self.reconstruction_type == 'primitive':
            self.dWdx = gradx
        elif self.reconstruction_type == 'conservative':
            self.dUdx = gradx
        else:
            raise ValueError('Reconstruction type ' + str(self.reconstruction_type) + ' is not defined.')

    @grady.setter
    def grady(self, grady):
        """
        Sets the y-direction gradients based on the reconstruction type. For example, if the reconstruction type is
        primitive, this will set the self.dWdy attribute.

        Parameters:
            - (np.ndarray): Values of the y-direction gradients.

        Return:
            - None
        """

        if self.reconstruction_type == 'primitive':
            self.dWdy = grady
        elif self.reconstruction_type == 'conservative':
            self.dUdy = grady
        else:
            raise ValueError('Reconstruction type ' + str(self.reconstruction_type) + ' is not defined.')

    def __getitem__(self, index):

        # Extract variables
        y, x, var = index

        if self._index_in_west_ghost_block(x, y):
            return self.ghost.W.state[y, 0, var]
        elif self._index_in_east_ghost_block(x, y):
            return self.ghost.E.state[y, 0, var]
        elif self._index_in_north_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        elif self._index_in_south_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        else:
            raise ValueError('Incorrect indexing')

    def _index_in_west_ghost_block(self, x, y):
        return x < 0 and 0 <= y <= self.mesh.ny

    def _index_in_east_ghost_block(self, x, y):
        return x > self.mesh.nx and 0 <= y <= self.mesh.ny

    def _index_in_south_ghost_block(self, x, y):
        return y < 0 and 0 <= x <= self.mesh.nx

    def _index_in_north_ghost_block(self, x, y):
        return y > self.mesh.ny and 0 <= x <= self.mesh.nx

    @staticmethod
    def _get_side_length(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    @staticmethod
    def _get_side_angle(pt1, pt2):
        if pt1[1] == pt2[1]:
            return np.pi / 2
        elif pt1[0] == pt2[0]:
            return 0
        else:
            return np.arctan((pt1[0] - pt2[0]) / (pt2[1] - pt1[1]))

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    def get_dt(self) -> np.float:
        """
        Return the time step for this block based on the CFL condition.

        Parameters:
            - None

        Returns:
            - dt (np.float): Float representing the value of the time step
        """

        # Speed of sound
        a = self.state.a()
        # Calculate dt using the CFL condition
        dt = self.inputs.CFL * np.amin(np.minimum(self.mesh.dx[:, :, 0] / (np.absolute(self.state.u) + a),
                                                  self.mesh.dy[:, :, 0] / (np.absolute(self.state.v) + a)))
        return dt

    def connect(self,
                NeighborE: QuadBlock,
                NeighborW: QuadBlock,
                NeighborN: QuadBlock,
                NeighborS: QuadBlock,
                NeighborNE: QuadBlock,
                NeighborNW: QuadBlock,
                NeighborSE: QuadBlock,
                NeighborSW: QuadBlock,
                ) -> None:
        """
        Create the Neighbors class used to set references to the neighbor blocks in each direction.

        Parameters:
            - None

        Return:
            - None
        """

        self.neighbors = Neighbors(E=NeighborE, W=NeighborW, N=NeighborN, S=NeighborS,
                                   NE=NeighborNE, NW=NeighborNW, SE=NeighborSE, SW=NeighborSW)

    def get_east_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the WEST boundary condition for this block's EAST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second last column on the block's state will be returned.
            - if nghost = 2, the second and third last column on the block's state will be returned.
            - general case, return -(nghost + 1):-1 columns

        This is illustrated in the figure below:

            nghost = 1: return second last column (-2)

                       ...        -3         -2         -1
                                       .............
            O----------O----------O----.-----O-----.----O
            |          |          |    .     |     .    |\
            |          |          |    .     |     .    |-O
            |          |          |    .     |     .    | |\
            O----------O----------O----.-----O-----.----O |-O
            |          |          |    .     |     .    |\| |\
            |          |          |    .     |     .    |-O |-O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
            |          |          |    .     |     .    |\| |\|
            |          |          |    .     |     .    |-O |-O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
            |          |          |    .     |     .    |\| |\|
            |          |          |    .     |     .    |-O | O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
             \|         \|         \|   .     \|    .    \| |\|
              O----------O----------O----.-----O-----.----O |-O
               \|         \|         \|   .     \|    .    \| |
                O----------O----------O----.-----O-----.----O |
                 \|         \|         \|   .     \|    .    \|
                  O----------O----------O----.-----O-----.----O
                                              .............

            nghost = 2: return third and second last colums (-3, -2)

                       ...        -3         -2         -1
                            ........................
            O----------O----.-----O----------O-----.----O
            |          |    .     |          |     .    |\
            |          |    .     |          |     .    |-O
            |          |    .     |          |     .    | |\
            O----------O----.-----O----------O-----.----O |-O
            |          |    .     |          |     .    |\| |\
            |          |    .     |          |     .    |-O |-O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
            |          |    .     |          |     .    |\| |\|
            |          |    .     |          |     .    |-O |-O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
            |          |    .     |          |     .    |\| |\|
            |          |    .     |          |     .    |-O | O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
             \|         \|   .     \|         \|    .    \| |\|
              O----------O----.-----O----------O-----.----O |-O
               \|         \|   .     \|         \|    .    \| |
                O----------O----.-----O----------O-----.----O |
                 \|         \|   .     \|         \|    .    \|
                  O----------O----.-----O----------O-----.----O
                                   ........................
        """

        return self.state.U[:, -self.mesh.nghost:, :].copy()

    def get_west_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the EAST boundary condition for this block's WEST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second column on the block's state will be returned.
            - if nghost = 2, the second and third column on the block's state will be returned.
            - general case, return 1:(nghost + 1) columns

        This is illustrated in the figure below:

            nghost = 1: return second column (1)

            0          1          2          3          4
                 ............
            O----.-----O----.-----O----------O----------O
            |    .     |    .     |          |          |\
            |    .     |    .     |          |          |-O
            |    .     |    .     |          |          | |\
            O----.-----O----.-----O----------O----------O |-O
            |    .     |    .     |          |          |\| |\
            |    .     |    .     |          |          |-O |-O
            |    .     |    .     |          |          | |\| |
            O----.-----O----.-----O----------O----------O |-O |
            |    .     |    .     |          |          |\| |\|
            |    .     |    .     |          |          |-O |-O
            |    .     |    .     |          |          | |\| |
            O----.-----O----.-----O----------O----------O |-O |
            |    .     |    .     |          |          |\| |\|
            |    .     |    .     |          |          |-O | O
            |    .     |    .     |          |          | |\| |
            O----.-----O----------O----------O----------O |-O |
             \|   .     \|   .     \|         \|         \| |\|
              O----.-----O----------O----------O----------O |-O
               \|   .     \|   .     \|         \|         \| |
                O----.-----O----------O----------O----------O |
                 \|   .     \|   .     \|         \|         \|
                  O----.-----O----.-----O----------O----------O
                        ............

            nghost = 2: return second and third colums (1, 2)

            0          1          2          3          4
                 .......................
            O----.-----O----------O----.-----O----------O
            |    .     |          |    .     |          |\
            |    .     |          |    .     |          |-O
            |    .     |          |    .     |          | |\
            O----.-----O----------O----.-----O----------O |-O
            |    .     |          |    .     |          |\| |\
            |    .     |          |    .     |          |-O |-O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
            |    .     |          |    .     |          |\| |\|
            |    .     |          |    .     |          |-O |-O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
            |    .     |          |    .     |          |\| |\|
            |    .     |          |    .     |          |-O | O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
             \|   .     \|         \|   .     \|         \| |\|
              O----.-----O----------O----.-----O----------O |-O
               \|   .     \|         \|   .     \|         \| |
                O----.-----O----------O----.-----O----------O |
                 \|   .     \|         \|   .     \|         \|
                  O----.-----O----------O----.-----O----------O
                        .......................
        """

        return self.state.U[:, :self.mesh.nghost, :].copy()

    def get_north_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the SOUTH boundary condition for this block's NORTH neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second last row on the block's state will be returned.
            - if nghost = 2, the second and third last rows on the block's state will be returned.
            - general case, return -(nghost + 1):-1 rows

        This is illustrated in the figure below:

            nghost = 1: return second last row (-2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
          ............................................... O
          . |          |          |          |          |.|\
          . O----------O----------O----------O----------O . O
          . |          |          |          |          | |.|\
          ............................................... O . O
            |          |          |          |          |.|\|.|
            O----------O----------O----------O----------O . O .
            |          |          |          |          |\|.|\|.
            |          |          |          |          |-O . O.
            |          |          |          |          | |\|.|.
            O----------O----------O----------O----------O |-O ..
            |          |          |          |          |\| |\|.
            |          |          |          |          |-O | O
            |          |          |          |          | |\| |
            O----------O----------O----------O----------O |-O |
             \|         \|         \|         \|         \| |\|
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O


            nghost = 2: return second and third last rows (-3, -2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
          ...............................................-O
          . |          |          |          |          |.|\
          . O----------O----------O----------O----------O . O
          . |          |          |          |          |\|.|\
          . |          |          |          |          |-O . O
          . |          |          |          |          | |\|.|
          . O----------O----------O----------O----------O | O .
          . |          |          |          |          |\| |\|.
          ...............................................-O | O.
            |          |          |          |          |.|\|.|.
            O----------O----------O----------O----------O .-O |.
            |          |          |          |          |\|.|\|.
            |          |          |          |          |-O . O.
            |          |          |          |          | |\|.|.
            O----------O----------O----------O----------O |-O ..
             \|         \|         \|         \|         \| |\|.
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O
        """

        return self.state.U[-self.mesh.nghost:, :, :].copy()

    def get_south_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the NORTH boundary condition for this block's SOUTH neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second row on the block's state will be returned.
            - if nghost = 2, the second and third rows on the block's state will be returned.
            - general case, return 1:(nghost + 1) rows

        This is illustrated in the figure below:

            nghost = 1: return second row (1)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
            |          |          |          |          |-O
            |          |          |          |          | |\
            O----------O----------O----------O----------O |-O
            |          |          |          |          |\| |\
            |          |          |          |          |-O |-O
            |          |          |          |          | |\| |
            O----------O----------O----------O----------O |-O |
            |          |          |          |          |\| |\|
          ...............................................-O |-O
          . |          |          |          |          |.|\| |
          . O----------O----------O----------O----------O .-O |
          . |          |          |          |          |\|.|\|
          ...............................................-O .-O
            |          |          |          |          |.|\|.|
            O----------O----------O----------O----------O .-O .
             \|         \|         \|         \|         \|.|\|.
              O----------O----------O----------O----------O .-O.
               \|         \|         \|         \|         \|.|.
                O----------O----------O----------O----------O ..
                 \|         \|         \|         \|         \|.
                  O----------O----------O----------O----------O


            nghost = 2: return second and third rows (1, 2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
            |          |          |          |          |-O
            |          |          |          |          | |\
            O----------O----------O----------O----------O |-O
            |          |          |          |          | | |\
          ...............................................-O |-O
          . |          |          |          |          |.|\| |
          . O----------O----------O----------O----------O .-O |
          . |          |          |          |          |\|.|\|
          . |          |          |          |          |-O .-O
          . |          |          |          |          | |\|.|
          . O----------O----------O----------O----------O |-O .
          . |          |          |          |          |\| |\|.
          ...............................................-O |-O.
            |          |          |          |          |.|\| |.
            O----------O----------O----------O----------O .-O |.
             \|         \|         \|         \|         \|.|\|.
              O----------O----------O----------O----------O .-O.
               \|         \|         \|         \|         \|.|.
                O----------O----------O----------O----------O ..
                 \|         \|         \|         \|         \|.
                  O----------O----------O----------O----------O
        """

        return self.state.U[:self.mesh.nghost, :, :].copy()

    def get_east_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the east edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: ny * 1 * 4 numpy ndarray with the east edge data

        """

        return self.state.U[:, -1:, :].copy()

    def get_west_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the west edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: ny * 1 * 4 numpy ndarray with the west edge data

        """

        return self.state.U[:, 0:1, :].copy()

    def get_north_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the north edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: 1 * nx * 4 numpy ndarray with the north edge data

        """

        return self.state.U[-1:, :, :].copy()

    def get_south_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the south edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: 1 * nx * 4 numpy ndarray with the south edge data

        """

        return self.state.U[0:1, :, :].copy()

    def row(self,
            index: int,
            copy: bool = False
            ) -> np.ndarray:
        """
        Return the solution stored in the index-th row of the mesh. For example, if index is 0, then the state at the
        most-bottom row of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which row needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - (np.ndarray): The numpy array containing the solution at the index-th row being returned.
        """
        _row = self.state.U[index, None, :, :]
        return _row.copy() if copy else _row

    def fullrow(self,
                index: int,
                copy: bool = False
                ) -> np.ndarray:
        """
        Return the solution stored in the index-th full row of the mesh. A full row is defined as the row plus the ghost
        cells on either side of the column.

        Parameters:
            - index (int): The index that reperesents which full row needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - (np.ndarray): The numpy array containing the solution at the index-th full row being returned.
        """
        _fullrow = np.concatenate((self.ghost.W[index, None, :, :],
                                   self.row(index),
                                   self.ghost.E[index, None, :, :]),
                                  axis=1)
        return _fullrow.copy() if copy else _fullrow

    def col(self,
            index: int,
            copy: bool = False
            ) -> np.ndarray:
        """
        Return the solution stored in the index-th column of the mesh. For example, if index is 0, then the state at the
        left-most column of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which column needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - (np.ndarray): The numpy array containing the soution at the index-th column being returned.
        """
        _col = self.state.U[None, :, index, :]
        return _col.copy() if copy else _col

    def fullcol(self,
                index: int,
                copy: bool = False
                ) -> np.ndarray:
        """
        Return the solution stored in the index-th full column of the mesh. A full column is defined as the row plus the
        ghost cells on either side of the column.

        Parameters:
            - index (int): The index that reperesents which full column needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - (np.ndarray): The numpy array containing the solution at the index-th full column being returned.
        """
        _fullcol = np.concatenate((self.ghost.S[:, index, None, :],
                                   self.col(index),
                                   self.ghost.N[:, index, None, :]),
                                  axis=1)
        return _fullcol.copy() if copy else _fullcol

    def get_interface_values(self) -> [np.ndarray]:

        if self.inputs.interface_interpolation == 'arithmetic_average':
            interfaceE, interfaceW, interfaceN, interfaceS = self.get_interface_values_arithmetic()
            #interfaceEW, interfaceNS = self.get_interface_values_arithmetic()
            return interfaceE, interfaceW, interfaceN, interfaceS
        else:
            raise ValueError('Interface Interpolation method is not defined.')

    def get_interface_values_arithmetic(self) -> [np.ndarray]:

        catx = np.concatenate((self.ghost.W.state.Q,
                                self.state.Q,
                                self.ghost.E.state.Q),
                                axis=1)

        caty = np.concatenate((self.ghost.S.state.Q,
                                self.state.Q,
                                self.ghost.N.state.Q),
                                axis=0)

        # Compute arithmetic mean
        interfaceEW = 0.5 * (catx[:, 1:, :] + catx[:, :-1, :])
        interfaceNS = 0.5 * (caty[1:, :, :] + caty[:-1, :, :])

        return interfaceEW[:, 1:, :], interfaceEW[:, :-1, :], interfaceNS[1:, :, :], interfaceNS[:-1, :, :]
        #return interfaceEW, interfaceNS

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    def update(self, dt: float) -> None:
        """
        Updates the solution state stored in the current block. Also includes any pre-processing needed prior to the
        calculation of the state updates, such as preconditioning, etc. The core operation in this method is calling the
        _time_integrator class, which steps the solution state through time by an amount of dt seconds.

        Parameters:
            - dt (float): Time step

        Returns:
            - N.A
        """
        self._time_integrator(self, dt)
        if not self.realizable():
            raise ValueError('Negative or zero pressure, density, or energy. Terminating simulation.')

    def get_flux(self) -> None:
        """
        Calls the get_flux() method from the Block's finite-volume-method to compute the flux at each cell wall.

        Parameters:
            - None

        Returns:
            - None
        """
        self.fvm.get_flux(self)

    def dUdt(self) -> np.ndarray:
        """
        Calls the dUdt() method from the Block's finite-volume-method to compute the residuals used for the time
        marching scheme.

        Parameters:
            - None

        Returns:
            - None
        """

        return self.fvm.dUdt(self)

    def set_BC(self) -> None:
        """
        Calls the set_BC() method for each ghost block connected to this block. This sets the boundary condition on
        each side.corner of the block.

        Parameters:
            - None

        Returns:
            - None
        """
        self.ghost.E.set_BC()
        self.ghost.W.set_BC()
        self.ghost.N.set_BC()
        self.ghost.S.set_BC()

    # ------------------------------------------------------------------------------------------------------------------
    # Gradient methods

    def drho_dx(self) -> np.ndarray:
        """
        Calculate the derivative of density with respect to the x direction. This is equivalent of returning
        gradx[:, :, 0], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - drho_dx(np.ndarray): Derivative of rho with respect to the x direction.
        """

        return self.gradx[:, :, 0, None]

    def du_dx(self) -> np.ndarray:
        """
        Calculate the derivative of u with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.gradx`. To compute this gradient, utilize the chain rule on drhou_dx:

        \\frac{\\partial(\\rho u)}{\\partial x} = \\rho \\frac{\\partial u}{\\partial x} +
        u \\frac{\\partial \\rho}{\\partial x},

        and rearrange to compute du_dx:

        \\frac{\\partial u}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho u)}{\\partial x} -
        u \\frac{\\partial \\rho}{\\partial x} \\right).

        Parameters:
            - N.A

        Returns:
            - du_dx (np.ndarray): Derivative of u with respect to the x direction.
        """

        return (self.gradx[:, :, 1, None] - self.state.u * self.drho_dx()) / self.state.rho

    def dv_dx(self) -> np.ndarray:
        """
        Calculate the derivative of v with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.gradx`. To compute this gradient, utilize the chain rule on drhov_dx:

        \\frac{\\partial(\\rho v)}{\\partial x} = \\rho \\frac{\\partial v}{\\partial x} +
        v \\frac{\\partial \\rho}{\\partial x},

        and rearrange to compute dv_dx:

        \\frac{\\partial v}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho v)}{\\partial x} -
        v \\frac{\\partial \\rho}{\\partial x} \\right).

        Parameters:
            - N.A

        Returns:
            - dv_dx (np.ndarray): Derivative of v with respect to the x direction.
        """

        return (self.gradx[:, :, 2, None] - self.state.v * self.drho_dx()) / self.state.rho

    def de_dx(self) -> np.ndarray:
        """
        Calculate the derivative of energy with respect to the x direction. This is equivalent of returning
        gradx[:, :, 3], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - de_dx (np.ndarray): Derivative of e with respect to the x direction.
        """

        return self.gradx[:, :, -1, None]

    def drho_dy(self) -> np.ndarray:
        """
        Calculate the derivative of density with respect to the y direction. This is equivalent of returning
        grady[:, :, 0], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - drho_dy (np.ndarray): Derivative of rho with respect to the y direction.
        """
        return self.grady[:, :, 0, None]

    def du_dy(self) -> np.ndarray:
        """
        Calculate the derivative of u with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.grady`. To compute this gradient, utilize the chain rule on drhou_dy:

        \\frac{\\partial(\\rho u)}{\\partial y} = \\rho \\frac{\\partial u}{\\partial y} +
        u \\frac{\\partial \\rho}{\\partial y},

        and rearrange to compute du_dy:

        \\frac{\\partial u}{\\partial y} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho u)}{\\partial y} -
        u \\frac{\\partial \\rho}{\\partial y} \\right).

        Parameters:
            - N.A

        Returns:
            - du_dy (np.ndarray): Derivative of u with respect to the y direction.
        """

        return (self.grady[:, :, 1, None] - self.state.u * self.drho_dx()) / self.state.rho

    def dv_dy(self) -> np.ndarray:
        """
        Calculate the derivative of v with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.grady`. To compute this gradient, utilize the chain rule on drhov_dy:

        \\frac{\\partial(\\rho v)}{\\partial y} = \\rho \\frac{\\partial v}{\\partial y} +
        v \\frac{\\partial \\rho}{\\partial y},

        and rearrange to compute dv_dx:

        \\frac{\\partial v}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho v)}{\\partial y} -
        v \\frac{\\partial \\rho}{\\partial y} \\right).

        Parameters:
            - N.A

        Returns:
            - dv_dy (np.ndarray): Derivative of v with respect to the y direction.
        """

        return (self.grady[:, :, 2, None] - self.state.v * self.drho_dx()) / self.state.rho

    def de_dy(self) -> np.ndarray:
        """
        Calculate the derivative of energy with respect to the y direction. This is equivalent of returning
        grady[:, :, 3], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - de_dy (np.ndarray): Derivative of e with respect to the y direction.
        """

        return self.grady[:, :, -1, None]


    def get_nodal_solution(self,
                           interpolation: str = 'piecewise_linear',
                           formulation: str = 'primitive',
                           ) -> np.ndarray:

        if interpolation == 'piecewise_linear':

            if formulation == 'primitive':
                return self._get_nodal_solution_piecewise_linear_primitive()
            elif formulation == 'conservative':
                return self._get_nodal_solution_piecewise_linear_conservative()
            else:
                raise ValueError('Formulation ' + str(interpolation) + 'is not defined.')

        elif interpolation == 'cell_average':

            if formulation == 'primitive':
                return self._get_nodal_solution_cell_average_primitive()
            elif formulation == 'conservative':
                return self._get_nodal_solution_cell_average_conservative()
            else:
                raise ValueError('Formulation ' + str(interpolation) + 'is not defined.')

        else:
            raise ValueError('Interpolation method ' + str(interpolation) + 'has not been specialized.')


    def _get_nodal_solution_piecewise_linear_primitive(self) -> np.ndarray:
        pass


    def _get_nodal_solution_piecewise_linear_conservative(self) -> np.ndarray:
        pass


    def _get_nodal_solution_cell_average_primitive(self) -> np.ndarray:
        pass


    def _get_nodal_solution_cell_average_conservative(self) -> np.ndarray:

        # Initialize solution array
        U = np.zeros((self.inputs.ny + 1, self.inputs.nx + 1, 4), dtype=float)

        # Set corners

        # South-West
        U[0, 0, :] = self.state.U[0, 0, :]

        # North-West
        U[0, -1, :] = self.state.U[0, -1, :]

        # South-East
        U[-1, 0, :] = self.state.U[-1, 0, :]

        # North-East
        U[-1, -1, :] = self.state.U[-1, -1, :]

        # East edge
        U[1:-1, -1, :] = 0.5 * (self.state.U[1:, -1, :] + self.state.U[:-1, -1, :])
        # West edge
        U[1:-1, 0, :] = 0.5 * (self.state.U[1:, 0, :] + self.state.U[:-1, 0, :])
        # North edge
        if self.neighbors.N:
            U[-1, 1:-1, :] = 0.25 * (self.state.U[-1, 1:, :] +
                                     self.state.U[-1, :-1, :] +
                                     self.neighbors.N.state.U[0, 1:, :] +
                                     self.neighbors.N.state.U[0, :-1, :])
        else:
            U[-1, 1:-1, :] = 0.5 * (self.state.U[-1, 1:, :] +
                                     self.state.U[-1, :-1, :])
        # South edge
        if self.neighbors.S:
            U[0, 1:-1, :] = 0.25 * (self.state.U[0, 1:, :] +
                                    self.state.U[0, :-1, :] +
                                    self.neighbors.S.state.U[-1, 1:, :] +
                                    self.neighbors.S.state.U[-1, :-1, :])
        else:
            U[0, 1:-1, :] = 0.5 * (self.state.U[0, 1:, :] +
                                   self.state.U[0, :-1, :])

        # Kernel
        U[1:-1, 1:-1, :] = 0.25 * (self.state.U[1:, 1:, :] +
                                   self.state.U[:-1, :-1, :] +
                                   self.state.U[1:, :-1, :] +
                                   self.state.U[:-1, 1:, :])

        return U

    def realizable(self):
        return self.state.realizable() and all([blk.realizable() for blk in self.ghost()])
