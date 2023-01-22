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
import functools
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Callable, Union, Type

import numba as nb
import matplotlib.pyplot as plt

from pyhype.utils.utils import (
    NumpySlice,
    CornerPropertyContainer,
    FullPropertyContainer,
)
from pyhype.flux import FluxFunctionFactory
from pyhype.limiters import SlopeLimiterFactory
from pyhype.fvm import FiniteVolumeMethodFactory
from pyhype.gradients import GradientFactory
from pyhype.blocks import quad_block as qb
from pyhype.states import PrimitiveState, ConservativeState

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.mesh.quad_mesh import QuadMesh
    from pyhype.solvers.base import SolverConfig
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.mesh.quadratures import QuadraturePoint
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

    :cvar ALL_IDX: A numpy slice across all dimensions.
    :ivar use_JIT: Flag that controls the use of JIT compiled
    functions.
    """

    ALL_IDX = np.s_[:, :, :]

    def __init__(self, config):
        self.use_JIT = config.use_JIT

    @abstractmethod
    def get_high_order_term(
        self,
        x_c: np.ndarray,
        x_p: np.ndarray,
        y_c: np.ndarray,
        y_p: np.ndarray,
        slicer: slice or tuple or int = None,
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
        slicer: slice or tuple or int = SolutionGradients.ALL_IDX,
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
                for k in range(g_x.shape[2]):
                    term[i, j, k] = g_x[i, j, k] * (x_p[i, j, 0] - x_c[i, j, 0]) + g_y[
                        i, j, k
                    ] * (y_p[i, j, 0] - y_c[i, j, 0])
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
        slicer: slice or tuple or int = None,
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


class BlockMixin:

    state: State

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
        return self.state[index, None, :, :]

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
        return self.state[None, :, index, :]


class Blocks:
    """
    Class for containing the solution blocks and necessary methods.

    :ivar config: SolverConfigs object
    :ivar num_BLK: Number of blocks
    :ivar blocks: Dict for containing the QuadBlock objects
    :ivar cpu: (future use) indicates the rank of the CPU responsible
    for this collection of blocks.
    """

    def __init__(self, config) -> None:
        self.config = config
        self.num_BLK = None
        self.blocks = {}
        self.cpu = None

        self.build()

    def __getitem__(self, blknum: int) -> QuadBlock:
        """
        Returns a specific SolutionBlock based on its number.

        :param blknum: Number of the SolutionBlock to return

        :return: SolutionBlock object
        """
        return self.blocks[blknum]

    def add(self, block: QuadBlock) -> None:
        """
        Adds a block to the dict that contains the SolutionBlocks

        :param block: SolutionBlock to add.

        :return: None
        """
        self.blocks[block.global_nBLK] = block

    def update(self, dt: float) -> None:
        """
        Applies the time marching procedure to the blocks.

        :param dt: Time step

        :return: None
        """
        for block in self.blocks.values():
            block.update(dt)

    def apply_boundary_condition(self) -> None:
        """
        Applies the boundary conditions to every block.

        :return: None
        """
        for block in self.blocks.values():
            block.apply_boundary_condition()

    def build(self) -> None:
        """
        Builds the SolutionBlocks contained in this Blocks collection.

        :return: None
        """
        self.num_BLK = len(self.config.mesh)
        for blk_data in self.config.mesh.values():
            self.add(qb.QuadBlock(self.config, blk_data))

        for block in self.blocks.values():
            neighbors = self.config.mesh[block.global_nBLK].neighbors
            block.connect(
                NeighborE=self.blocks[neighbors.E] if neighbors.E is not None else None,
                NeighborW=self.blocks[neighbors.W] if neighbors.W is not None else None,
                NeighborN=self.blocks[neighbors.N] if neighbors.N is not None else None,
                NeighborS=self.blocks[neighbors.S] if neighbors.S is not None else None,
                NeighborNE=None,
                NeighborNW=None,
                NeighborSE=None,
                NeighborSW=None,
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


class BaseBlockState(BaseBlockMesh, BlockMixin):
    """
    Block object that inherits from BaseBlockMesh and BlockMixin,
    which also contains the solution State and relevant methods.

    :cvar EAST_BOUND_IDX: East block boundary numpy indexing slice.
    :cvar WEST_BOUND_IDX: West block boundary numpy indexing slice.
    :cvar NORTH_BOUND_IDX: North block boundary numpy indexing slice.
    :cvar SOUTH_BOUND_IDX: South block boundary numpy indexing slice.

    :cvar EAST_FACE_IDX: East cell faces numpy indexing slice.
    :cvar WEST_FACE_IDX: West cell faces numpy indexing slice.
    :cvar NORTH_FACE_IDX: North cell faces numpy indexing slice.
    :cvar SOUTH_FACE_IDX: South cell faces numpy indexing slice.

    :ivar state: State object of the appropriate type.
    """

    EAST_BOUND_IDX = NumpySlice.east_boundary()
    WEST_BOUND_IDX = NumpySlice.west_boundary()
    NORTH_BOUND_IDX = NumpySlice.north_boundary()
    SOUTH_BOUND_IDX = NumpySlice.south_boundary()

    EAST_FACE_IDX = NumpySlice.east_face()
    WEST_FACE_IDX = NumpySlice.west_face()
    NORTH_FACE_IDX = NumpySlice.north_face()
    SOUTH_FACE_IDX = NumpySlice.south_face()

    def __init__(
        self,
        config: SolverConfig,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(config=config, mesh=mesh, qp=qp)
        self.state = state_type(fluid=config.fluid, shape=(mesh.ny, mesh.nx, 4))


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
        slicer: slice or tuple or int = None,
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

        flux = FluxFunctionFactory.get(
            config=config, type=self.config.fvm_flux_function_type
        )
        gradient = GradientFactory.get(
            config=config, type=self.config.fvm_gradient_type
        )
        limiter = SlopeLimiterFactory.get(
            config=config, type=config.fvm_slope_limiter_type
        )
        self.fvm = FiniteVolumeMethodFactory.create(
            config=config,
            type=config.fvm_type,
            order=config.fvm_spatial_order,
            flux=flux,
            limiter=limiter,
            gradient=gradient,
        )

    def unlimited_reconstruction_at_location(
        self,
        x_c: np.ndarray,
        y_c: np.ndarray,
        x_p: np.ndarray,
        y_p: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> State:
        """
        Computes the unlimited reconstruction of the solution state at
        given x and y coordinates.

        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :param slicer: Array slicing object

        :return: State object with the unlimited reconstruction at the
        given x and y coordinates.
        """
        state = self.state if slicer is None else self.state[slicer]
        return state + self.high_order_term_at_location(x_c, x_p, y_c, y_p, slicer)

    def limited_reconstruction_at_location(
        self,
        x_c: np.ndarray,
        y_c: np.ndarray,
        x_p: np.ndarray,
        y_p: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> State:
        """
        Computes the limited reconstruction of the solution state at
        given x and y coordinates.
        :param x_c: x-coordinates of cell centroids
        :param x_p: x-coordinates of quadrature point
        :param y_c: y-coordinates of cell centroids
        :param y_p: y-coordinates of quadrature point
        :param slicer: Array slicing object

        :return: State object with the limited reconstruction at the
        given x and y coordinates.
        """
        high_order_term = self.high_order_term_at_location(x_c, x_p, y_c, y_p, slicer)
        if slicer is None:
            return self.state + self.fvm.limiter.phi * high_order_term
        return self.state[slicer] + self.fvm.limiter.phi[slicer] * high_order_term

    def get_east_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's east
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the east block boundary quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.E, slicer=self.EAST_BOUND_IDX
        )

    def get_west_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's west
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the west block boundary quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.W, slicer=self.WEST_BOUND_IDX
        )

    def get_north_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's north
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the north block boundary quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.N, slicer=self.NORTH_BOUND_IDX
        )

    def get_south_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's south
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the south block boundary quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.S, slicer=self.SOUTH_BOUND_IDX
        )

    def get_east_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's east
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the east face quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.E, slicer=self.EAST_FACE_IDX
        )

    def get_west_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's west
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the west face quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.W, slicer=self.WEST_FACE_IDX
        )

    def get_north_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's north
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the north face quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.N, slicer=self.NORTH_FACE_IDX
        )

    def get_south_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's south
        boundary at each quadrature point.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the south face quadrature points.
        """
        return self.get_limited_recon_states_at_qp(
            qps=self.qp.S, slicer=self.SOUTH_FACE_IDX
        )

    def get_limited_recon_states_at_qp(
        self, slicer: slice, qps: tuple[QuadraturePoint]
    ) -> tuple[np.ndarray]:
        """
        Return the limited solution reconstruction at specified quadrature points
        along a specified slice of the arrays.

        :param slicer: Numpy Slice object
        :param qps: QuadraturePointData that indicates where the solution is to
        be reconstructed.

        :rtype: tuple[State]
        :return: tuple of states containing the limited solution reconstructions
        at the south face quadrature points.
        """
        states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state,
                refBLK=self,
                qp=qp,
                slicer=slicer,
            )
            for qp in qps
        )
        return states


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
        self.vertices = CornerPropertyContainer(NE=NE, NW=NW, SE=SE, SW=SW)
        self.n = nx * ny
        self.nx = nx
        self.ny = ny
        self.nghost = nghost


class BlockInfo:
    def __init__(self, blk_input: dict):
        # Set parameter attributes from input dict
        self.nBLK = blk_input["nBLK"]

        self.neighbors = FullPropertyContainer(
            E=blk_input["NeighborE"],
            W=blk_input["NeighborW"],
            N=blk_input["NeighborN"],
            S=blk_input["NeighborS"],
            NE=blk_input["NeighborNE"],
            NW=blk_input["NeighborNW"],
            SE=blk_input["NeighborSE"],
            SW=blk_input["NeighborSW"],
        )
        self.bc = FullPropertyContainer(
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
