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
from itertools import chain
from typing import TYPE_CHECKING, Type, Union, List

import numba as nb
import matplotlib.pyplot as plt
import mpi4py as mpi
from matplotlib.collections import LineCollection

from pyhype.mesh import quadratures
from pyhype.mesh.quad_mesh import QuadMesh
from pyhype.blocks.ghost import GhostBlocks
from pyhype.flux import FluxFunctionFactory
from pyhype.gradients import GradientFactory
from pyhype.limiters import SlopeLimiterFactory
from pyhype.fvm import FiniteVolumeMethodFactory
from pyhype.states.conservative import ConservativeState
from pyhype.utils.utils import NumpySlice, SidePropertyDict
from pyhype.blocks.base import BaseBlockFVM, BlockDescription, ExtraProcessNeighborInfo

from pyhype.utils.logger import Logger

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.blocks.ghost import GhostBlock
    from pyhype.solvers.base import SolverConfig
    from pyhype.mesh.quadratures import QuadraturePointData

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class BaseBlockGhost(BaseBlockFVM):
    def __init__(
        self,
        config: SolverConfig,
        block_data: BlockDescription,
        parent_block: Union[QuadBlock, ReconstructionBlock],
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ) -> None:
        """
        Constructs instance of class BaseBlock_With_Ghost.

        :type config: SolverConfigs
        :param config: Object that contains all the input parameters that decribe the problem.

        :type block_data: BlockDescription
        :param block_data: Object containing the parameters that describe the block

        :type parent_block: BaseBlockGhost
        :param parent_block: Reference to the interior block that the ghost cells need to store

        :type state_type: str
        :param state_type: Type of the state in the block and the ghost blocks

        :return: None
        """
        self.neighbors = None
        self.block_data = block_data
        self.global_block_num = block_data.info.nBLK

        super().__init__(
            config,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

        self.cpu = mpi.MPI.COMM_WORLD.Get_rank()
        self.ghost = GhostBlocks(
            config=config,
            block_data=block_data,
            parent_block=parent_block,
            state_type=state_type,
        )
        self.fvm = FiniteVolumeMethodFactory.create(
            config=config,
            flux=FluxFunctionFactory.create(config=config),
            limiter=SlopeLimiterFactory.create(config=config),
            gradient=GradientFactory.create(config=config),
            parent_block=self,
        )

        self.EAST_GHOST_IDX = NumpySlice.cols(-self.config.nghost, None)
        self.WEST_GHOST_IDX = NumpySlice.cols(None, self.config.nghost)
        self.NORTH_GHOST_IDX = NumpySlice.rows(-self.config.nghost, None)
        self.SOUTH_GHOST_IDX = NumpySlice.rows(None, self.config.nghost)

        self.is_cartesian = self._is_cartesian()

    def ghost_bc_type(self, direction: int):
        """
        Get the boundary condition type associated with the given direction

        :param direction: Direction to check BC type
        :return: bc type
        """
        return self.ghost[direction].bc_type

    def ghost_block(self, direction: int) -> GhostBlock:
        """
        Return reference to the ghost block in the specified direction

        :param direction: Direction to get the ghost block for
        :return: Ghost block in that direction
        """
        return self.ghost[direction]

    def _is_cartesian(self) -> bool:
        """
        Return boolen value that indicates if the block is alligned with the cartesian axes.

        Parameters:
            - None

        Return:
            - is_cartesian (bool): Boolean that is True if the block is cartesian and False if it isnt
        """

        is_cartesian = (
            (self.mesh.vertices.NE[1] == self.mesh.vertices.NW[1])
            and (self.mesh.vertices.SE[1] == self.mesh.vertices.SW[1])
            and (self.mesh.vertices.SE[0] == self.mesh.vertices.NE[0])
            and (self.mesh.vertices.SW[0] == self.mesh.vertices.NW[0])
        )
        return is_cartesian

    def realizable(self):
        realizable = [self.state.realizable()]
        realizable.extend(blk.realizable() for blk in self.ghost.get_blocks())
        return realizable

    def from_block(self, from_block: BaseBlockGhost) -> None:
        """
        Updates the state in the interior and ghost blocks, which may be any subclasss of `State`, using the state in
        the interior and ghost blocks from the input block, which is of type `PrimitiveState`.

        :type from_block: BaseBlock_With_Ghost
        :param from_block: Block whos interior and ghost block states are used to update self.

        :return: None
        """
        self.state.from_state(from_block.state)
        for gblk, gblk_from in zip(
            self.ghost.get_blocks(), from_block.ghost.get_blocks()
        ):
            gblk.state.from_state(gblk_from.state)

    def get_interface_values(self) -> [np.ndarray]:
        """
        Compute values at the midpoint of the cell interfaces. The method of computing the values is specified in the
        input file, and its respective implementation must be included in the class.

        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        :return: Arrays containing interface values on each face
        """
        if self.config.interface_interpolation == "arithmetic_average":
            (
                interfaceE,
                interfaceW,
                interfaceN,
                interfaceS,
            ) = self.get_interface_values_arithmetic()
        else:
            raise ValueError("Interface Interpolation method is not defined.")

        return interfaceE, interfaceW, interfaceN, interfaceS

    def get_interface_values_arithmetic(self) -> [np.ndarray]:
        """
        Compute the midpoint interface values via an arithmetic mean of the state values in the cells on either side
        of each interface.

        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        :return: Numpy arrays containing the arithmetic mean-based interface values
        """

        # Concatenate ghost cell and state values in the East-West and North-South directions
        interfaceEW, interfaceNS = self._get_interface_values_arithmetic_JIT(
            self.ghost.E.state.data,
            self.ghost.W.state.data,
            self.ghost.N.state.data,
            self.ghost.S.state.data,
            self.state.data,
        )

        return (
            interfaceEW[:, 1:, :],
            interfaceEW[:, :-1, :],
            interfaceNS[1:, :, :],
            interfaceNS[:-1, :, :],
        )

    @staticmethod
    @nb.njit(cache=True)
    def _get_interface_values_arithmetic_JIT(
        ghostE,
        ghostW,
        ghostN,
        ghostS,
        state,
    ):
        shape = state.shape
        interfaceEW = np.zeros((shape[0], shape[1] + 1, 4))
        interfaceNS = np.zeros((shape[0] + 1, shape[1], 4))

        # East-West faces
        for i in range(state.shape[0]):
            for k in range(state.shape[2]):
                interfaceEW[i, 0, k] = 0.5 * (ghostW[i, 0, k] + state[i, 0, k])
                interfaceEW[i, -1, k] = 0.5 * (ghostE[i, 0, k] + state[i, -1, k])

        # North-South faces
        for j in range(state.shape[1]):
            for k in range(state.shape[2]):
                interfaceNS[0, j, k] = 0.5 * (ghostS[0, j, k] + state[0, j, k])
                interfaceNS[-1, j, k] = 0.5 * (ghostN[0, j, k] + state[-1, j, k])

        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                for k in range(state.shape[2]):
                    if j > 0:
                        interfaceEW[i, j, k] = 0.5 * (
                            state[i, j - 1, k] + state[i, j, k]
                        )
                    if i > 0:
                        interfaceNS[i, j, k] = 0.5 * (
                            state[i - 1, j, k] + state[i, j, k]
                        )

        return interfaceEW, interfaceNS

    def clear_cache(self):
        self.state.clear_cache()
        self.ghost.clear_cache()

    def get_dt(self) -> np.float:
        """
        Return the time step for this block based on the CFL condition.

        Parameters:
            - None

        Returns:
            - dt (np.float): Float representing the value of the time step
        """
        a = self.state.a()
        tx = self.mesh.dx[:, :, 0] / (np.absolute(self.state.u) + a)
        ty = self.mesh.dy[:, :, 0] / (np.absolute(self.state.v) + a)
        return self.config.CFL * np.amin(np.minimum(tx, ty))

    def connect(
        self,
        NeighborE: QuadBlock,
        NeighborW: QuadBlock,
        NeighborN: QuadBlock,
        NeighborS: QuadBlock,
    ) -> None:
        """
        Create the Neighbors class used to set references to the neighbor blocks in each direction.

        Parameters:
            - None

        Return:
            - None
        """
        self.neighbors = SidePropertyDict(
            E=NeighborE,
            W=NeighborW,
            N=NeighborN,
            S=NeighborS,
        )

    def neighbor_in_this_process(self, direction: int):
        """
        Returns True if the neighbor in the specified direction exists in the current process
        :param direction: Direction of neighbor (east, west, etc...)
        :return: If the neighbor is in the current process
        """
        return not isinstance(self.neighbors[direction], ExtraProcessNeighborInfo)

    def get_east_ghost_states(self) -> State:
        """
        Return the solution data used to build the WEST boundary condition for this block's EAST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:
            - if nghost = 1, the second last column on the block's state will be returned.
            - if nghost = 2, the second and third last column on the block's state will be returned.
            - general case, return -(nghost + 1):-1 columns
        """
        return self.state[self.EAST_GHOST_IDX]

    def get_west_ghost_states(self) -> State:
        """
        Return the solution data used to build the EAST boundary condition for this block's WEST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:
            - if nghost = 1, the second column on the block's state will be returned.
            - if nghost = 2, the second and third column on the block's state will be returned.
            - general case, return 1:(nghost + 1) columns
        """
        return self.state[self.WEST_GHOST_IDX]

    def get_north_ghost_states(self) -> State:
        """
        Return the solution data used to build the SOUTH boundary condition for this block's NORTH neighbor. The shape
        of the required data is dependent on the number of ghost blocks selected in the input file (nghost).
        For example:
            - if nghost = 1, the second last row on the block's state will be returned.
            - if nghost = 2, the second and third last rows on the block's state will be returned.
            - general case, return -(nghost + 1):-1 rows
        """
        return self.state[self.NORTH_GHOST_IDX]

    def get_south_ghost_states(self) -> State:
        """
        Return the solution data used to build the NORTH boundary condition for this block's SOUTH neighbor. The shape
        of the required data is dependent on the number of ghost blocks selected in the input file (nghost).
        For example:
            - if nghost = 1, the second row on the block's state will be returned.
            - if nghost = 2, the second and third rows on the block's state will be returned.
            - general case, return 1:(nghost + 1) rows
        """
        return self.state[self.SOUTH_GHOST_IDX]

    def send_boundary_data(self) -> List[mpi.MPI.Request]:
        """
        Sends boundary data to the appropriate location, which could be its own ghost blocks,
        a neighbor's ghost blocks on the same process, or a neighbors ghost blocks on a
        different process via MPI

        :return: List of active MPI send requests
        """
        send_reqs = [ghost.send_boundary_data() for ghost in self.ghost.values()]
        return [req for req in send_reqs if req is not None]

    def recieve_boundary_data(self) -> List[mpi.MPI.Request]:
        """
        Recieves boundary data from neighbors who communicated with MPI

        :return: List of active MPI recieve requests
        """
        recv_reqs = [ghost.recieve_boundary_data() for ghost in self.ghost.values()]
        return [req for req in recv_reqs if req is not None]

    def apply_data_buffers(self) -> None:
        """
        Applies the recieved data buffers via MPI to the state

        :return: None
        """
        for ghost in self.ghost.values():
            ghost.apply_recv_buffers_to_state()

    def apply_boundary_condition(self) -> None:
        """
        Calls the apply_boundary_condition() method for each ghost block connected to this block. This sets the boundary condition on
        each side.corner of the block.

        :return: None
        """
        for ghost in self.ghost.values():
            ghost.apply_boundary_condition()

    def get_nodal_solution(
        self,
        interpolation: str = "piecewise_linear",
        formulation: str = "primitive",
    ) -> np.ndarray:

        if interpolation == "piecewise_linear":

            if formulation == "primitive":
                return self._get_nodal_solution_piecewise_linear_primitive()
            if formulation == "conservative":
                return self._get_nodal_solution_piecewise_linear_conservative()
            raise ValueError("Formulation " + str(interpolation) + "is not defined.")

        if interpolation == "cell_average":

            if formulation == "primitive":
                return self._get_nodal_solution_cell_average_primitive()
            if formulation == "conservative":
                return self._get_nodal_solution_cell_average_conservative()
            raise ValueError("Formulation " + str(interpolation) + "is not defined.")
        raise ValueError(
            "Interpolation method " + str(interpolation) + "has not been specialized."
        )

    def _get_nodal_solution_piecewise_linear_primitive(self) -> np.ndarray:
        pass

    def _get_nodal_solution_piecewise_linear_conservative(self) -> np.ndarray:
        pass

    def _get_nodal_solution_cell_average_primitive(self) -> np.ndarray:
        pass

    def _get_nodal_solution_cell_average_conservative(self) -> np.ndarray:

        # Initialize solution array
        U = np.zeros((self.config.ny + 1, self.config.nx + 1, 4), dtype=float)

        # Set corners

        # South-West
        U[0, 0, :] = self.state.data[0, 0, :]

        # North-West
        U[0, -1, :] = self.state.data[0, -1, :]

        # South-East
        U[-1, 0, :] = self.state.data[-1, 0, :]

        # North-East
        U[-1, -1, :] = self.state.data[-1, -1, :]

        # East edge
        U[1:-1, -1, :] = 0.5 * (
            self.state.data[1:, -1, :] + self.state.data[:-1, -1, :]
        )
        # West edge
        U[1:-1, 0, :] = 0.5 * (self.state.data[1:, 0, :] + self.state.data[:-1, 0, :])
        # North edge
        if self.neighbors.N:
            U[-1, 1:-1, :] = 0.25 * (
                self.state.data[-1, 1:, :]
                + self.state.data[-1, :-1, :]
                + self.neighbors.N.state.data[0, 1:, :]
                + self.neighbors.N.state.data[0, :-1, :]
            )
        else:
            U[-1, 1:-1, :] = 0.5 * (
                self.state.data[-1, 1:, :] + self.state.data[-1, :-1, :]
            )
        # South edge
        if self.neighbors.S:
            U[0, 1:-1, :] = 0.25 * (
                self.state.data[0, 1:, :]
                + self.state.data[0, :-1, :]
                + self.neighbors.S.state.data[-1, 1:, :]
                + self.neighbors.S.state.data[-1, :-1, :]
            )
        else:
            U[0, 1:-1, :] = 0.5 * (
                self.state.data[0, 1:, :] + self.state.data[0, :-1, :]
            )

        # Kernel
        U[1:-1, 1:-1, :] = 0.25 * (
            self.state.data[1:, 1:, :]
            + self.state.data[:-1, :-1, :]
            + self.state.data[1:, :-1, :]
            + self.state.data[:-1, 1:, :]
        )

        return U


class ReconstructionBlock(BaseBlockGhost):
    def __init__(
        self,
        config: SolverConfig,
        block_data: BlockDescription,
        mesh: QuadMesh,
        qp: QuadraturePointData,
    ) -> None:
        super().__init__(
            config,
            block_data=block_data,
            parent_block=self,
            mesh=mesh,
            qp=qp,
            state_type=config.reconstruction_type,
        )


class QuadBlock(BaseBlockGhost):
    def __init__(
        self,
        config: SolverConfig,
        block_data: BlockDescription,
    ) -> None:
        self.block_data = block_data
        self.global_block_num = block_data.info.nBLK
        self.mpi = mpi.MPI.COMM_WORLD
        mesh = QuadMesh(config, block_data.geometry)
        qp = quadratures.QuadraturePointData(config, refMESH=mesh)
        super().__init__(
            config,
            block_data=block_data,
            mesh=mesh,
            qp=qp,
            parent_block=self,
            state_type=ConservativeState,
        )
        self.recon_block = ReconstructionBlock(
            config,
            block_data,
            qp=qp,
            mesh=mesh,
        )
        self._logger = Logger(config=config)

    @property
    def reconstruction_type(self):
        """
        Returns the reconstruction type used in the finite volume method.

        Parameters:
            - None

        Return:
            - (str): the reconstruction type
        """

        return type(self.recon_block.state)

    def get_updated_recon_block(self) -> BaseBlockGhost:
        if self.config.reconstruction_type == type(self):
            return self
        self.recon_block.from_block(self)
        return self.recon_block
