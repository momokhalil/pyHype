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
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Union, Type

import mpi4py as mpi
import numba as nb
import matplotlib.pyplot as plt

from pyhype.utils.utils import (
    NumpySlice,
    SidePropertyDict,
    FullPropertyDict,
    CornerPropertyDict,
)
from pyhype.blocks import quad_block as qb
from pyhype.flux import FluxFunctionFactory
from pyhype.gradients import GradientFactory
from pyhype.limiters import SlopeLimiterFactory
from pyhype.fvm import FiniteVolumeMethodFactory
from pyhype.states import PrimitiveState, ConservativeState

from pyhype.utils.logger import Logger

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.mesh.quad_mesh import QuadMesh
    from pyhype.solvers.base import SolverConfig
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.mesh.quadratures import QuadraturePointData

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class Neighbors:
    """
    A class that holds references to a Block's neighbors.

    :ivar E: Reference to the east neighbor
    :ivar W: Reference to the west neighbor
    :ivar N: Reference to the north neighbor
    :ivar S: Reference to the south neighbor
    :ivar NE: Reference to the north-east neighbor
    :ivar NW: Reference to the north-west neighbor
    :ivar SE: Reference to the south-east neighbor
    :ivar SW: Reference to the south-west neighbor
    """

    def __init__(
        self,
        E: QuadBlock = None,
        W: QuadBlock = None,
        N: QuadBlock = None,
        S: QuadBlock = None,
        NE: QuadBlock = None,
        NW: QuadBlock = None,
        SE: QuadBlock = None,
        SW: QuadBlock = None,
    ) -> None:
        """
        Save the references to the Block's neighbors.

        :type E: QuadBlock
        :param E: East neighbor block

        :type W: QuadBlock
        :param W: West neighbor block

        :type N: QuadBlock
        :param N: North neighbor block

        :type S: QuadBlock
        :param S: South neighbor block

        :type NE: QuadBlock
        :param NE: North-East neighbor block

        :type NW: QuadBlock
        :param NW: North-West neighbor block

        :type SE: QuadBlock
        :param SE: South-East neighbor block

        :type SW: QuadBlock
        :param SW: South-West neighbor block

        :rtype: None
        :return: None
        """
        self.E = E
        self.W = W
        self.N = N
        self.S = S
        self.NE = NE
        self.NW = NW
        self.SE = SE
        self.SW = SW


class NormalVector:
    """
    A class that holds the x- and y-components of a normal vector,
    calculated based on a given angle theta. The angle theta is defined
    anti-clockwise starting from the east direction as follows:

                            North
                              ^
                              |    /
                              |   /*
                              |  /   *
                              | /      *
                              |/ theta  *
              <------------------------------> East

    :ivar x: x-component of the normal vector
    :ivar y: x-component of the normal vector
    """

    def __init__(self, theta: float) -> None:
        """
        Instantiates the class and calculates the x- and y-components
        based on the given angle theta.

        :type theta: float
        :param theta: angle in radians

        :rtype: None
        :return: None
        """
        if theta == 0:
            self.x, self.y = 1, 0
        elif theta == np.pi / 2:
            self.x, self.y = 0, 1
        elif theta == np.pi:
            self.x, self.y = -1, 0
        elif theta == 3 * np.pi / 2:
            self.x, self.y = 0, -1
        elif theta == 2 * np.pi:
            self.x, self.y = 1, 0
        else:
            self.x = np.cos(theta)
            self.y = np.sin(theta)

    def __str__(self) -> str:
        """
        Print type, including the values of x and y.

        :rtype: str
        :return: String that decribes the class and provides the values of x and y
        """
        return "NormalVector object: [" + str(self.x) + ", " + str(self.y) + "]"


class SolutionGradients(ABC):
    """
    Class that contains the solution gradients and performs
    basic operations with them.

    :cvar all_slice: A numpy slice across all dimensions.
    :ivar use_JIT: Flag that controls the use of JIT compiled
    functions.
    """

    all_slice = np.s_[:, :, :]

    def __init__(self, config):
        self.use_JIT = config.use_JIT

    @abstractmethod
    def get_high_order_term(
        self,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
        slicer: Union[slice, tuple, int] = None,
    ) -> np.ndarray:
        """
        Calculates and returns an array containing the high-order term
        for a high-order reconstruction method.

        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :param slicer: Array slicing object

        :raise NotImplementedError: Raises NotImplementedError since this
        is an abstract class.
        """
        raise NotImplementedError(
            "Calling get_high_order_term() from the SolutionGradients ABC."
        )


class FirstOrderGradients(SolutionGradients):
    """
    Class that contains the first order solution gradients and performs
    basic operations with them.

    :ivar x: Numpy array containing the x-direction first-order gradients.
    :ivar y: Numpy array containing the y-direction first-order gradients.
    """

    def __init__(self, config, nx: int, ny: int):
        super().__init__(config)
        self.x = np.zeros(shape=(ny, nx, 4))
        self.y = np.zeros(shape=(ny, nx, 4))

    def get_high_order_term(
        self,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
        slicer: Union[slice, tuple, int] = SolutionGradients.all_slice,
    ) -> np.ndarray:
        """
        Calculates and returns an array containing the first-order gradient
        terms for a high-order reconstruction method.

        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :param slicer: Array slicing object
        :return: High order term array
        """
        if self.use_JIT:
            return self.get_high_order_term_JIT(
                self.x[slicer],
                self.y[slicer],
                x_c[slicer],
                x_p[slicer],
                y_c[slicer],
                y_p[slicer],
            )
        return self.x[slicer] * (x_p[slicer] - x_c[slicer]) + self.y[slicer] * (
            y_p[slicer] - y_c[slicer]
        )

    @staticmethod
    @nb.njit(cache=True)
    def get_high_order_term_JIT(
        g_x: np.ndarray,
        g_y: np.ndarray,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
    ) -> np.ndarray:
        """
        JIT implementation of the high-order term calculation.

        :param g_x: x-direction gradient
        :param g_y: y-direction gradient
        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :return: High order term array
        """
        term = np.zeros_like(g_x)
        for i in range(g_x.shape[0]):
            for j in range(g_x.shape[1]):
                dx = x_p[i, j, 0] - x_c[i, j, 0]
                dy = y_p[i, j, 0] - y_c[i, j, 0]
                for k in range(g_x.shape[2]):
                    term[i, j, k] = g_x[i, j, k] * dx + g_y[i, j, k] * dy
        return term


class SecondOrderGradients(SolutionGradients):
    """
    Class that contains the second order solution gradients and performs
    basic operations with them.

    :ivar x: Numpy array containing the x-direction first-order gradients.
    :ivar y: Numpy array containing the y-direction first-order gradients.
    :ivar xx: Numpy array containing the x-direction second-order gradients.
    :ivar yy: Numpy array containing the y-direction second-order gradients.
    :ivar xy: Numpy array containing the x,y-direction second-order gradients.
    """

    def __init__(self, config, nx: int, ny: int):
        super().__init__(config)
        self.x = np.zeros(shape=(ny, nx, 4))
        self.y = np.zeros(shape=(ny, nx, 4))
        self.xx = np.zeros(shape=(ny, nx, 4))
        self.yy = np.zeros(shape=(ny, nx, 4))
        self.xy = np.zeros(shape=(ny, nx, 4))

    def get_high_order_term(
        self,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
        slicer: Union[slice, tuple, int] = None,
    ) -> np.ndarray:
        """
        Calculates and returns an array containing the second-order gradient
        terms for a high-order reconstruction method.

        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :param slicer: Array slicing object
        :return: High order term array
        """
        if slicer is None:
            x = x_p - x_c
            y = y_p - y_c
            return (
                self.x * x
                + self.y * y
                + self.xx * x**2
                + self.yy * y**2
                + self.xy * x * y
            )

        x = x_p[slicer] - x_c[slicer]
        y = y_p[slicer] - y_c[slicer]
        return (
            self.x[slicer] * x
            + self.y[slicer] * y
            + self.xy[slicer] * x * y
            + self.xx[slicer] * x**2
            + self.yy[slicer] * y**2
        )


class GradientsFactory:
    @staticmethod
    def create(
        config, reconstruction_order: int, nx: int, ny: int
    ) -> SolutionGradients:
        """
        Factory for creating solution gradients appropriate for the given
        reconstruction order input.

        :param config: Problem config object
        :param reconstruction_order: The order of the reconstruction methods
        :param nx: Number of cells in the x-direction
        :param ny: Number of cells in the y-direction

        :return: SolutionGradients object of the appropriate order
        """
        if reconstruction_order == 2:
            return FirstOrderGradients(config, nx, ny)
        if reconstruction_order == 4:
            return SecondOrderGradients(config, nx, ny)
        raise ValueError(
            "GradientsFactory.create_gradients(): Error, no gradients container class has been "
            "extended for the given order."
        )


class Blocks:
    """
    Class for containing the solution blocks and necessary methods.

    :ivar config: SolverConfigs object
    :ivar blocks: Dict for containing the QuadBlock objects
    :ivar cpu: (future use) indicates the rank of the CPU responsible
    for this collection of blocks.
    """

    def __init__(
        self,
        config: SolverConfig,
        mesh_config: dict,
    ) -> None:
        self._logger = Logger(config=config)

        self.config = config
        self.mesh_config = mesh_config
        self.blocks = {}
        self.mpi = mpi.MPI.COMM_WORLD
        self.cpu = self.mpi.Get_rank()

        self.build()

    @property
    def num_blocks(self):
        return self.__len__()

    def __iter__(self):
        return self.blocks.items()

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, blknum: int) -> QuadBlock:
        """
        Returns a specific SolutionBlock based on its number.

        :param blknum: Number of the SolutionBlock to return

        :return: SolutionBlock object
        """
        return self.blocks[blknum]

    def add(self, blocks: [QuadBlock]) -> None:
        """
        Adds a block to the dict that contains the SolutionBlocks

        :param blocks: Iterable of QuadBlock to add.

        :return: None
        """
        for block in blocks:
            self.blocks[block.global_nBLK] = block

    def apply_boundary_condition(self) -> None:
        """
        Applies the boundary conditions to every block.

        :return: None
        """
        reqs = []

        for block in self.blocks.values():
            reqs.extend(block.send_boundary_data())
            reqs.extend(block.recieve_boundary_data())

        try:
            mpi.MPI.Request.Waitall(reqs)
        except mpi.MPI.Exception:
            error_code = mpi.MPI.Status().Get_error()
            self._logger.info(error_code)

        for block in self.blocks.values():
            block.apply_data_buffers()

        for block in self.blocks.values():
            block.apply_boundary_condition()

    @staticmethod
    def distribute_blocks_to_processes(
        num_blocks: int,
        num_processes: int,
    ) -> dict[int, int]:
        """
        Distributes num_blocks blocks to num_processes processes. Every process will have a
        baseline number of blocks, and some will have one extra if num_blocks is not divisible
        by num_processes.

        Example:
            num_blocks: 8 (numbered 0, 1, 2, 3, 4, 5, 6, 7)
            num_processes: 3 (numbered 0, 1, 2)

            then:
            num_full_processes = 2 (will have baseline + 1 block)
            blocks_per_process_baseline = 2

            then the processes will contain blocks:
            process 0: [0, 1, 2] (baseline + 1 num blocks)
            process 1: [3, 4, 5] (baseline + 1 num blocks)
            process 2: [6, 7] (baseline num blocks)

        :param num_blocks: Number of blocks to distribute
        :param num_processes: Number of processes to distribute blocks into
        :return: dict that maps {block_num: process_num}
        """
        cpus = {}
        counter = 0
        num_full_processes = num_blocks % num_processes
        blocks_per_process_baseline = num_blocks // num_processes

        for n in range(num_processes):
            local_num_blocks = (
                blocks_per_process_baseline + 1
                if n < num_full_processes
                else blocks_per_process_baseline
            )
            cpus.update({i: n for i in [counter + i for i in range(local_num_blocks)]})
            counter += local_num_blocks
        return cpus

    def build(self) -> None:
        """
        Builds the SolutionBlocks contained in this Blocks collection.

        :return: None
        """
        cpu_dict = self.distribute_blocks_to_processes(
            num_blocks=len(self.mesh_config),
            num_processes=self.mpi.Get_size(),
        )
        blocks_in_this_proccess = [
            block_num for block_num, cpu_num in cpu_dict.items() if cpu_num == self.cpu
        ]
        self.add(
            qb.QuadBlock(
                self.config,
                block_data=self.mesh_config[block_num],
            )
            for block_num in blocks_in_this_proccess
        )

        def get_neighbor_ref(neighbor_num: int) -> Union[int, QuadBlock]:
            """
            Returns a valid reference to the neighbor block with the specified number.
            If the number is on shared memory (process running on the same machine),
            the reference to the neighbor's QuadBlock object is returned. If it is on a
            separate node (non-shared memory), a tuple is returned containing the block's
            number and process number. If the neighbor number indidated is None, that means
            that there is no neighbor in that direction, and None is returned.

            :param neighbor_num: Global block number for neighbor
            :return: Either a block/process number or a QuadBlock reference to the neighbor
            if it exists, else None.
            """
            return (
                (
                    (neighbor_num, cpu_dict[neighbor_num])
                    if neighbor_num not in self.blocks
                    else self.blocks[neighbor_num]
                )
                if neighbor_num is not None
                else None
            )

        for block in self.blocks.values():
            ne, nw, nn, ns = (
                get_neighbor_ref(neigh)
                for neigh in self.mesh_config[block.global_nBLK].neighbors.values()
            )
            block.connect(
                NeighborE=ne,
                NeighborW=nw,
                NeighborN=nn,
                NeighborS=ns,
            )

    def print_connectivity(self) -> None:
        """
        Prints the connectivity pattern for each SolutionBlock.

        :return: None
        """
        for _, block in self.blocks.items():
            print("-----------------------------------------")
            print(
                "CONNECTIVITY FOR GLOBAL BLOCK: ",
                block.global_nBLK,
                "<{}>".format(block),
            )
            print("North: ", block.neighbors.N)
            print("South: ", block.neighbors.S)
            print("East:  ", block.neighbors.E)
            print("West:  ", block.neighbors.W)

    def plot_mesh(self):
        """
        Plots the mesh of each SolutionBlock.

        :return: None
        """
        _, ax = plt.subplots(1)
        ax.set_aspect("equal")
        for block in self.blocks.values():
            block.plot(ax=ax)
        plt.show()
        plt.pause(0.001)
        plt.close()


class BaseBlock:
    def __init__(self, config: SolverConfig):
        self.config = config

    @staticmethod
    def _is_all_blk_conservative(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, ConservativeState), blks))

    @staticmethod
    def _is_all_blk_primitive(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, PrimitiveState), blks))


class BaseBlockMesh(BaseBlock):
    """
    BaseBlock class that contains a mesh and quadrature point object.
    This is the basic building block for all block types.

    :ivar config: SolverConfigs object
    :ivar mesh: QuadMesh object containing the block's mesh
    :ivar qp: QuadraturePointData object that contains the quadrature point
    locations and methods for quadrature point calculations.
    """

    def __init__(
        self,
        config: SolverConfig,
        mesh: QuadMesh = None,
        qp: QuadraturePointData = None,
    ):
        super().__init__(config=config)
        self.mesh = mesh
        self.qp = qp


class BaseBlockState(BaseBlockMesh):
    """
    Block object that inherits from BaseBlockMesh,
    which also contains the solution State and relevant methods.

    :cvar east_boundary_slice: East block boundary numpy indexing slice.
    :cvar west_boundary_slice: West block boundary numpy indexing slice.
    :cvar north_boundary_slice: North block boundary numpy indexing slice.
    :cvar south_boundary_slice: South block boundary numpy indexing slice.

    :cvar east_face_slice: East cell faces numpy indexing slice.
    :cvar west_face_slice: West cell faces numpy indexing slice.
    :cvar north_face_slice: North cell faces numpy indexing slice.
    :cvar south_face_slice: South cell faces numpy indexing slice.

    :ivar state: State object of the appropriate type.
    """

    east_boundary_slice = NumpySlice.east_boundary()
    west_boundary_slice = NumpySlice.west_boundary()
    north_boundary_slice = NumpySlice.north_boundary()
    south_boundary_slice = NumpySlice.south_boundary()

    east_face_slice = NumpySlice.east_face()
    west_face_slice = NumpySlice.west_face()
    north_face_slice = NumpySlice.north_face()
    south_face_slice = NumpySlice.south_face()

    def __init__(
        self,
        config: SolverConfig,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(config=config, mesh=mesh, qp=qp)
        self.state = state_type(fluid=config.fluid, shape=(mesh.ny, mesh.nx, 4))

    def row(self, index: Union[int, slice]) -> State:
        """
        Return the solution stored in the index-th row of the mesh.
        For example, if index is 0, then the state at the most-bottom
        row of the mesh will be returned.

        :param index: The index that reperesents which row needs
        to be returned.

        :return: Numpy array containing the solution at the index-th
        row being returned.
        """
        return self.state[NumpySlice.row(index=index)]

    def col(self, index: int) -> State:
        """
        Return the solution stored in the index-th column of the mesh.
        For example, if index is 0, then the state at the left-most column
        of the mesh will be returned.

        :param index: The index that reperesents which column needs to
        be returned.

        :return: The numpy array containing the soution at the index-th
        column being returned.
        """
        return self.state[NumpySlice.col(index=index)]


class BaseBlockGrad(BaseBlockState):
    """
    Class that inherits from BaseBlockState and adds a SolutionGradient
    object.

    :ivar grad: SolutionGradients object
    """

    def __init__(
        self,
        config: SolverConfig,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(config, mesh=mesh, qp=qp, state_type=state_type)
        self.grad = GradientsFactory.create(
            config=config,
            reconstruction_order=config.fvm_spatial_order,
            nx=mesh.nx,
            ny=mesh.ny,
        )

    def high_order_term_at_location(
        self,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
        slicer: Union[slice, tuple, int] = None,
    ) -> np.ndarray:
        return self.grad.get_high_order_term(x_c, x_p, y_c, y_p, slicer)


class BaseBlockFVM(BaseBlockGrad):
    """
    Class that inherits from BaseBlockGrad and adds a finite volume method
    object and approriate functions.

    :ivar fvm: Finite volume method object
    """

    def __init__(
        self,
        config: SolverConfig,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(config, mesh=mesh, qp=qp, state_type=state_type)

        self.fvm = FiniteVolumeMethodFactory.create(
            config=config,
            flux=FluxFunctionFactory.create(config=config),
            limiter=SlopeLimiterFactory.create(config=config),
            gradient=GradientFactory.create(config=config),
            parent_block=self,
        )


class BlockGeometry:
    """ """

    def __init__(
        self,
        NE: [float] = None,
        NW: [float] = None,
        SE: [float] = None,
        SW: [float] = None,
        nx: int = None,
        ny: int = None,
        nghost: int = None,
    ):
        self.vertices = CornerPropertyDict(NE=NE, NW=NW, SE=SE, SW=SW)
        self.n = nx * ny
        self.nx = nx
        self.ny = ny
        self.nghost = nghost


class BlockInfo:
    def __init__(self, blk_input: dict):
        # Set parameter attributes from input dict
        self.nBLK = blk_input["nBLK"]

        self.neighbors = SidePropertyDict(
            E=blk_input["NeighborE"],
            W=blk_input["NeighborW"],
            N=blk_input["NeighborN"],
            S=blk_input["NeighborS"],
        )
        self.bc = FullPropertyDict(
            E=blk_input["BCTypeE"],
            W=blk_input["BCTypeW"],
            N=blk_input["BCTypeN"],
            S=blk_input["BCTypeS"],
            NE=blk_input["BCTypeNE"],
            NW=blk_input["BCTypeNW"],
            SE=blk_input["BCTypeSE"],
            SW=blk_input["BCTypeSW"],
        )


class BlockDescription:
    def __init__(self, nx: int, ny: int, blk_input: dict = None, nghost: int = None):
        # Set parameter attributes from input dict
        self.info = None
        if blk_input is not None:
            self.info = BlockInfo(blk_input=blk_input)

        self.geometry = BlockGeometry(
            NE=blk_input["NE"],
            NW=blk_input["NW"],
            SE=blk_input["SE"],
            SW=blk_input["SW"],
            nx=nx,
            ny=ny,
            nghost=nghost,
        )

    @property
    def neighbors(self):
        return self.info.neighbors

    @property
    def bc(self):
        return self.info.bc
