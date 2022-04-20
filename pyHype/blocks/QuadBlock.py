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
from itertools import chain

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pyHype.fvm import FirstOrderMUSCL, SecondOrderMUSCL
from pyHype.mesh.QuadMesh import QuadMesh
from pyHype.mesh import quadratures as qp
from pyHype.mesh.base import BlockDescription
from pyHype.utils.utils import NumpySlice
from pyHype.blocks.base import Neighbors, BaseBlock, BaseBlock_Only_State
from pyHype.solvers.time_integration.explicit_runge_kutta import ExplicitRungeKutta as Erk
from pyHype.blocks.ghost import GhostBlocks

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput

class GradientsContainer:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    def check_gradients(self):
        pass

class FirstOrderGradients(GradientsContainer):
    def __init__(self, inputs: ProblemInput):
        super().__init__(inputs)
        self.x = np.empty(shape=(inputs.ny, inputs.nx, 4))
        self.y = np.empty(shape=(inputs.ny, inputs.nx, 4))

class SecondOrderGradients(GradientsContainer):
    def __init__(self, inputs: ProblemInput):
        super().__init__(inputs)
        self.x = np.empty(shape=(inputs.ny, inputs.nx, 4))
        self.y = np.empty(shape=(inputs.ny, inputs.nx, 4))
        self.xx = np.empty(shape=(inputs.ny, inputs.nx, 4))
        self.yy = np.empty(shape=(inputs.ny, inputs.nx, 4))
        self.xy = np.empty(shape=(inputs.ny, inputs.nx, 4))

class GradientsFactory:
    @staticmethod
    def create_gradients(inputs: ProblemInput):
        if inputs.fvm_spatial_order == 1:
            return FirstOrderGradients(inputs)
        if inputs.fvm_spatial_order == 2:
            return SecondOrderGradients(inputs)
        raise ValueError('GradientsFactory.create_gradients(): Error, no gradients container class has been '
                         'extended for the given order.')

class BaseBlock_With_Ghost(BaseBlock_Only_State):

    def __init__(self, inputs: ProblemInput, nx: int, ny: int, block_data: BlockDescription, refBLK: BaseBlock,
                 state_type: str = 'conservative') -> None:
        """
        Constructs instance of class BaseBlock_With_Ghost.

        :type inputs: ProblemInputs
        :param inputs: Object that contains all the input parameters that decribe the problem.

        :type nx: int
        :param nx: Number of cells in the x direction

        :type nx: int
        :param ny: Number of cells in the y direction

        :type block_data: BlockDescription
        :param block_data: Object containing the parameters that describe the block

        :type refBLK: QuadBlock
        :param refBLK: Reference to the interior block that the ghost cells need to store

        :type state_type: str
        :param state_type: Type of the state in the block and the ghost blocks

        :return: None
        """
        super().__init__(inputs, nx, ny, state_type=state_type)
        self.ghost = GhostBlocks(inputs, block_data, refBLK=refBLK, state_type=state_type)

    def to_primitive(self) -> None:
        """
        Converts state in the interior and ghost blocks to `PrimitiveState`.

        :return: None
        """
        self.state = self.state.to_primitive_state()
        for gblk in self.ghost():
            gblk.state = gblk.state.to_primitive_state()

    def to_conservative(self) -> None:
        """
        Converts state in the interior and ghost blocks to `ConservativeState`.

        :return: None
        """
        self.state = self.state.to_conservative_state()
        for gblk in self.ghost():
            gblk.state = gblk.state.to_conservative_state()

    def from_primitive(self, from_block: BaseBlock_With_Ghost) -> None:
        """
        Updates the state in the interior and ghost blocks, which may be any subclasss of `State`, using the state in
        the interior and ghost blocks from the input block, which is of type `PrimitiveState`.

        :type from_block: BaseBlock_With_Ghost
        :param from_block: Block whos interior and ghost block states are used to update self.

        :return: None
        """
        self.state.from_primitive_state(from_block.state)
        for gblk, gblk_from in zip(self.ghost(), from_block.ghost()):
            gblk.state.from_primitive_state(gblk_from.state)

    def from_conservative(self, from_block: BaseBlock_With_Ghost) -> None:
        """
        Updates the state in the interior and ghost blocks, which may be any subclasss of `State`, using the state in
        the interior and ghost blocks from the input block, which is of type `ConservativeState`.

        :type from_block: BaseBlock_With_Ghost
        :param from_block: Block whos interior and ghost block states are used to update self.

        :return: None
        """
        self.state.from_conservative_state(from_block.state)
        for gblk, gblk_from in zip(self.ghost(), from_block.ghost()):
            gblk.state.from_conservative_state(gblk_from.state)

    def get_interface_values(self) -> [np.ndarray]:
        """
        Compute values at the midpoint of the cell interfaces. The method of computing the values is specified in the
        input file, and its respective implementation must be included in the class.

        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        :return: Arrays containing interface values on each face
        """
        if self.inputs.interface_interpolation == 'arithmetic_average':
            interfaceE, interfaceW, interfaceN, interfaceS = self.get_interface_values_arithmetic()
        else:
            raise ValueError('Interface Interpolation method is not defined.')

        return interfaceE, interfaceW, interfaceN, interfaceS

    def get_interface_values_arithmetic(self) -> [np.ndarray]:
        """
        Compute the midpoint interface values via an arithmetic mean of the state values in the cells on either side
        of each interface.

        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        :return: Numpy arrays containing the arithmetic mean-based interface values
        """

        # Concatenate ghost cell and state values in the East-West and North-South directions
        catx = np.concatenate((self.ghost.W.state.Q, self.state.Q, self.ghost.E.state.Q), axis=1)
        caty = np.concatenate((self.ghost.S.state.Q, self.state.Q, self.ghost.N.state.Q), axis=0)

        # Compute arithmetic mean
        interfaceEW = 0.5 * (catx[:, 1:, :] + catx[:, :-1, :])
        interfaceNS = 0.5 * (caty[1:, :, :] + caty[:-1, :, :])

        return interfaceEW[:, 1:, :], interfaceEW[:, :-1, :], interfaceNS[1:, :, :], interfaceNS[:-1, :, :]


class QuadBlock(BaseBlock_With_Ghost):
    EAST_FACE_IDX = NumpySlice.east_boundary()
    WEST_FACE_IDX = NumpySlice.west_boundary()
    NORTH_FACE_IDX = NumpySlice.north_boundary()
    SOUTH_FACE_IDX = NumpySlice.south_boundary()

    def __init__(self,
                 inputs: ProblemInput,
                 block_data: BlockDescription
                 ) -> None:

        self.inputs             = inputs
        self.block_data         = block_data
        self.mesh               = QuadMesh(inputs, block_data)
        self.global_nBLK        = block_data.nBLK
        self.ghost              = None
        self.neighbors          = None
        self.nx                 = inputs.nx
        self.ny                 = inputs.ny
        self.grad               = GradientsFactory.create_gradients(inputs)

        self.EAST_GHOST_IDX = NumpySlice.cols(-self.inputs.nghost, None)
        self.WEST_GHOST_IDX = NumpySlice.cols(None, self.inputs.nghost)
        self.NORTH_GHOST_IDX = NumpySlice.rows(-self.inputs.nghost, None)
        self.SOUTH_GHOST_IDX = NumpySlice.rows(None, self.inputs.nghost)

        super().__init__(inputs, nx=inputs.nx, ny=inputs.ny, block_data=block_data, refBLK=self)

        # Create reconstruction block
        self.reconBlk = BaseBlock_With_Ghost(inputs, inputs.nx, inputs.ny, block_data=block_data, refBLK=self,
                                             state_type=self.inputs.reconstruction_type)
        # Set finite volume method
        if self.inputs.fvm_type == 'MUSCL':
            if self.inputs.fvm_spatial_order == 1:
                self.fvm = FirstOrderMUSCL(self.inputs, self.global_nBLK)
            elif self.inputs.fvm_spatial_order == 2:
                self.fvm = SecondOrderMUSCL(self.inputs, self.global_nBLK)
            else:
                raise ValueError('No MUSCL finite volume method has been specialized with order '
                                 + str(self.inputs.fvm_spatial_order))
        else:
            raise ValueError('Specified finite volume method has not been specialized.')

        # Set time integrator
        if self.inputs.time_integrator   == 'ExplicitEuler1':
            self._time_integrator   = Erk.ExplicitEuler1(self.inputs)
        elif self.inputs.time_integrator == 'RK2':
            self._time_integrator   = Erk.RK2(self.inputs)
        elif self.inputs.time_integrator == 'Generic2':
            self._time_integrator   = Erk.Generic2(self.inputs)
        elif self.inputs.time_integrator == 'Ralston2':
            self._time_integrator   = Erk.Ralston2(self.inputs)
        elif self.inputs.time_integrator == 'Generic3':
            self._time_integrator   = Erk.Generic3(self.inputs)
        elif self.inputs.time_integrator == 'RK3':
            self._time_integrator   = Erk.RK3(self.inputs)
        elif self.inputs.time_integrator == 'RK3SSP':
            self._time_integrator   = Erk.RK3SSP(self.inputs)
        elif self.inputs.time_integrator == 'Ralston3':
            self._time_integrator   = Erk.Ralston3(self.inputs)
        elif self.inputs.time_integrator == 'RK4':
            self._time_integrator   = Erk.RK4(self.inputs)
        elif self.inputs.time_integrator == 'Ralston4':
            self._time_integrator   = Erk.Ralston4(self.inputs)
        elif self.inputs.time_integrator == 'DormandPrince5':
            self._time_integrator   = Erk.DormandPrince5(self.inputs)
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # is block cartesian
        self.is_cartesian = self._is_cartesian()

        # Quadrature
        self.QP = qp.QuadraturePointData(inputs, refMESH=self.mesh)


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

        _show = ax is None

        if not ax:
            _, ax = plt.subplots(1, 1)

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

    def __getitem__(self, index):

        # Extract variables
        y, x, var = index

        if self._index_in_west_ghost_block(x, y):
            return self.ghost.W.state[y, 0, var]
        if self._index_in_east_ghost_block(x, y):
            return self.ghost.E.state[y, 0, var]
        if self._index_in_north_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        if self._index_in_south_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        raise ValueError('Incorrect indexing')

    def _index_in_west_ghost_block(self, x, y):
        return x < 0 and 0 <= y <= self.mesh.ny

    def _index_in_east_ghost_block(self, x, y):
        return x > self.mesh.nx and 0 <= y <= self.mesh.ny

    def _index_in_south_ghost_block(self, x, y):
        return y < 0 and 0 <= x <= self.mesh.nx

    def _index_in_north_ghost_block(self, x, y):
        return y > self.mesh.ny and 0 <= x <= self.mesh.nx

    def get_dt(self) -> np.float:
        """
        Return the time step for this block based on the CFL condition.

        Parameters:
            - None

        Returns:
            - dt (np.float): Float representing the value of the time step
        """
        a = self.state.a()
        _tx = self.mesh.dx[:, :, 0] / (np.absolute(self.state.u) + a)
        _ty = self.mesh.dy[:, :, 0] / (np.absolute(self.state.v) + a)
        return self.inputs.CFL * np.amin(np.minimum(_tx, _ty))

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

    def get_east_boundary_states(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's east boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _east_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state,
                refBLK=self,
                qp=qpe,
                slicer=self.EAST_FACE_IDX)
            for qpe in self.QP.E)
        return _east_states

    def get_west_boundary_states(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's west boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _west_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state,
                refBLK=self,
                qp=qpw,
                slicer=self.WEST_FACE_IDX)
            for qpw in self.QP.W)
        return _west_states

    def get_north_boundary_states(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's north boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _north_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state,
                refBLK=self,
                qp=qpn,
                slicer=self.NORTH_FACE_IDX)
            for qpn in self.QP.N)
        return _north_states

    def get_south_boundary_states(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's south boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _south_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state,
                refBLK=self,
                qp=qps,
                slicer=self.SOUTH_FACE_IDX)
            for qps in self.QP.S)
        return _south_states

    def get_east_ghost_states(self) -> np.ndarray:
        """
        Return the solution data used to build the WEST boundary condition for this block's EAST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:
            - if nghost = 1, the second last column on the block's state will be returned.
            - if nghost = 2, the second and third last column on the block's state will be returned.
            - general case, return -(nghost + 1):-1 columns
        """
        return self.state.U[self.EAST_GHOST_IDX].copy()

    def get_west_ghost_states(self) -> np.ndarray:
        """
        Return the solution data used to build the EAST boundary condition for this block's WEST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:
            - if nghost = 1, the second column on the block's state will be returned.
            - if nghost = 2, the second and third column on the block's state will be returned.
            - general case, return 1:(nghost + 1) columns
        """
        return self.state.U[self.WEST_GHOST_IDX].copy()

    def get_north_ghost_states(self) -> np.ndarray:
        """
        Return the solution data used to build the SOUTH boundary condition for this block's NORTH neighbor. The shape
        of the required data is dependent on the number of ghost blocks selected in the input file (nghost).
        For example:
            - if nghost = 1, the second last row on the block's state will be returned.
            - if nghost = 2, the second and third last rows on the block's state will be returned.
            - general case, return -(nghost + 1):-1 rows
        """
        return self.state.U[self.NORTH_GHOST_IDX].copy()

    def get_south_ghost_states(self) -> np.ndarray:
        """
        Return the solution data used to build the NORTH boundary condition for this block's SOUTH neighbor. The shape
        of the required data is dependent on the number of ghost blocks selected in the input file (nghost).
        For example:
            - if nghost = 1, the second row on the block's state will be returned.
            - if nghost = 2, the second and third rows on the block's state will be returned.
            - general case, return 1:(nghost + 1) rows
        """
        return self.state.U[self.SOUTH_GHOST_IDX].copy()

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
            if formulation == 'conservative':
                return self._get_nodal_solution_piecewise_linear_conservative()
            raise ValueError('Formulation ' + str(interpolation) + 'is not defined.')

        if interpolation == 'cell_average':

            if formulation == 'primitive':
                return self._get_nodal_solution_cell_average_primitive()
            if formulation == 'conservative':
                return self._get_nodal_solution_cell_average_conservative()
            raise ValueError('Formulation ' + str(interpolation) + 'is not defined.')
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
        return self.state.realizable() and all(blk.realizable() for blk in self.ghost())
