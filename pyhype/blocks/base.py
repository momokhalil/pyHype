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
from abc import abstractmethod
from typing import TYPE_CHECKING, Callable, Union, Type

import numba as nb
import matplotlib.pyplot as plt

from pyhype import fvm
from pyhype.utils.utils import (
    NumpySlice,
    CornerPropertyContainer,
    FullPropertyContainer,
)
from pyhype.blocks import quad_block as qb
from pyhype.states import PrimitiveState, ConservativeState

if TYPE_CHECKING:
    from pyhype.mesh.quad_mesh import QuadMesh
    from pyhype.mesh.quadratures import QuadraturePointData, QuadraturePoint
    from pyhype.states.base import State
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.solvers.base import ProblemInput

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
    calculated based on a given angle theta.

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


class SolutionGradients:
    ALL_IDX = np.s_[:, :, :]

    def __init__(self, inputs):
        self.use_JIT = inputs.use_JIT

    def check_gradients(self):
        pass

    @abstractmethod
    def get_high_order_term(
        self,
        xc: np.ndarray,
        xp: np.ndarray,
        yc: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = None,
    ):
        return NotImplementedError

    def get_high_order_term_mesh_qp(
        self, mesh: QuadMesh, qp: QuadraturePoint, slicer: slice or tuple or int
    ):
        return self.get_high_order_term(mesh.x, qp.x, mesh.y, qp.y, slicer)


class FirstOrderGradients(SolutionGradients):
    def __init__(self, inputs, nx: int, ny: int):
        super().__init__(inputs)
        self.x = np.zeros(shape=(ny, nx, 4))
        self.y = np.zeros(shape=(ny, nx, 4))

    def get_high_order_term(
        self,
        xc: np.ndarray,
        xp: np.ndarray,
        yc: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = SolutionGradients.ALL_IDX,
    ) -> np.ndarray:
        if self.use_JIT:
            return self.get_high_order_term_JIT(
                self.x[slicer],
                self.y[slicer],
                xc[slicer],
                xp[slicer],
                yc[slicer],
                yp[slicer],
            )
        return self.x[slicer] * (xp[slicer] - xc[slicer]) + self.y[slicer] * (
            yp[slicer] - yc[slicer]
        )

    @staticmethod
    @nb.njit(cache=True)
    def get_high_order_term_JIT(
        gx: np.ndarray,
        gy: np.ndarray,
        xc: np.ndarray,
        xp: np.ndarray,
        yc: np.ndarray,
        yp: np.ndarray,
    ):
        term = np.zeros_like(gx)
        for i in range(gx.shape[0]):
            for j in range(gx.shape[1]):
                for k in range(gx.shape[2]):
                    term[i, j, k] = gx[i, j, k] * (xp[i, j, 0] - xc[i, j, 0]) + gy[
                        i, j, k
                    ] * (yp[i, j, 0] - yc[i, j, 0])
        return term


class SecondOrderGradients(SolutionGradients):
    def __init__(self, inputs, nx: int, ny: int):
        super().__init__(inputs)
        self.x = np.zeros(shape=(ny, nx, 4))
        self.y = np.zeros(shape=(ny, nx, 4))
        self.xx = np.zeros(shape=(ny, nx, 4))
        self.yy = np.zeros(shape=(ny, nx, 4))
        self.xy = np.zeros(shape=(ny, nx, 4))

    def get_high_order_term(
        self,
        xc: np.ndarray,
        xp: np.ndarray,
        yc: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        if slicer is None:
            x = xp - xc
            y = yp - yc
            return (
                self.x * x
                + self.y * y
                + self.xx * x**2
                + self.yy * y**2
                + self.xy * x * y
            )

        x = xp[slicer] - xc[slicer]
        y = yp[slicer] - yc[slicer]
        return (
            self.x[slicer] * x
            + self.y[slicer] * y
            + self.xy[slicer] * x * y
            + self.xx[slicer] * x**2
            + self.yy[slicer] * y**2
        )


class GradientsFactory:
    @staticmethod
    def create_gradients(
        inputs, order: int, nx: int, ny: int
    ) -> Union[FirstOrderGradients, SecondOrderGradients]:
        if order == 2:
            return FirstOrderGradients(inputs, nx, ny)
        if order == 4:
            return SecondOrderGradients(inputs, nx, ny)
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

        Parameters:
            - index (int): The index that reperesents which row needs
            to be returned.

        Return:
            - (np.ndarray): The numpy array containing the solution
            at the index-th row being returned.
        """
        return self.state[index, None, :, :]

    def col(self, index: int) -> State:
        """
        Return the solution stored in the index-th column of the mesh.
        For example, if index is 0, then the state at the left-most column
        of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which column
            needs to be returned.

        Return:
            - (np.ndarray): The numpy array containing the soution
            at the index-th column being returned.
        """
        return self.state[None, :, index, :]


class Blocks:
    def __init__(self, inputs) -> None:
        # Set inputs
        self.inputs = inputs
        # Number of blocks
        self.num_BLK = None
        # Blocks dictionary
        self.blocks = {}
        # cpu handling this list of blocks
        self.cpu = None

        # Build blocks
        self.build()

    @staticmethod
    def to_all_blocks(func: Callable):
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            for block in self.blocks.values():
                func(self, block, *args, **kwargs)

        return _wrapper

    def __getitem__(self, blknum: int) -> QuadBlock:
        return self.blocks[blknum]

    def add(self, block: QuadBlock) -> None:
        self.blocks[block.global_nBLK] = block

    def update(
        self,
        dt: float,
    ) -> None:
        for block in self.blocks.values():
            block.update(dt)

    def apply_boundary_condition(self) -> None:
        for block in self.blocks.values():
            block.apply_boundary_condition()

    def build(self) -> None:
        for blk_data in self.inputs.mesh.values():
            self.add(qb.QuadBlock(self.inputs, blk_data))

        self.num_BLK = len(self.blocks)

        for block in self.blocks.values():
            neighbors = self.inputs.mesh[block.global_nBLK].neighbors
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
        _, ax = plt.subplots(1)
        ax.set_aspect("equal")
        for block in self.blocks.values():
            block.plot(ax=ax)
        plt.show()
        plt.pause(0.001)
        plt.close()


class BaseBlock:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    @staticmethod
    def _is_all_blk_conservative(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, ConservativeState), blks))

    @staticmethod
    def _is_all_blk_primitive(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, PrimitiveState), blks))


class BaseBlockMesh(BaseBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        mesh: QuadMesh = None,
        qp: QuadraturePointData = None,
    ):
        super().__init__(inputs=inputs)
        self.mesh = mesh
        self.qp = qp


class BaseBlockState(BaseBlockMesh, BlockMixin):
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
        inputs: ProblemInput,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(inputs=inputs, mesh=mesh, qp=qp)
        self.state = state_type(fluid=inputs.fluid, shape=(mesh.ny, mesh.nx, 4))


class BaseBlockGrad(BaseBlockState):
    def __init__(
        self,
        inputs: ProblemInput,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(inputs, mesh=mesh, qp=qp, state_type=state_type)
        self.grad = GradientsFactory.create_gradients(
            inputs=inputs, order=inputs.fvm_spatial_order, nx=mesh.nx, ny=mesh.ny
        )

    def high_order_term_at_location(
        self,
        xc: np.ndarray,
        xp: np.ndarray,
        yc: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        return self.grad.get_high_order_term(xc, xp, yc, yp, slicer)


class BaseBlockFVM(BaseBlockGrad):
    def __init__(
        self,
        inputs: ProblemInput,
        mesh: QuadMesh,
        qp: QuadraturePointData,
        state_type: Type[State],
    ):
        super().__init__(inputs, mesh=mesh, qp=qp, state_type=state_type)

        # Set finite volume method
        if self.inputs.fvm_type == "MUSCL":
            if self.inputs.fvm_spatial_order == 1:
                self.fvm = fvm.FirstOrderMUSCL(self.inputs)
            elif self.inputs.fvm_spatial_order == 2:
                self.fvm = fvm.SecondOrderMUSCL(self.inputs)
            else:
                raise ValueError(
                    "No MUSCL finite volume method has been specialized with order "
                    + str(self.inputs.fvm_spatial_order)
                )
        else:
            raise ValueError("Specified finite volume method has not been specialized.")

    def unlimited_reconstruction_at_location(
        self,
        xc: np.ndarray,
        yc: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        if slicer is None:
            return self.state + self.high_order_term_at_location(xc, xp, yc, yp, slicer)
        return self.state[slicer] + self.high_order_term_at_location(
            xc, xp, yc, yp, slicer
        )

    def limited_reconstruction_at_location(
        self,
        xc: np.ndarray,
        yc: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        _high_order_term = self.high_order_term_at_location(xc, xp, yc, yp, slicer)
        if slicer is None:
            return self.state + self.fvm.limiter.phi * _high_order_term
        return self.state[slicer] + self.fvm.limiter.phi[slicer] * _high_order_term

    def get_east_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's east
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _east_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpe, slicer=self.EAST_BOUND_IDX
            )
            for qpe in self.qp.E
        )
        return _east_states

    def get_west_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's west
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _west_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpw, slicer=self.WEST_BOUND_IDX
            )
            for qpw in self.qp.W
        )
        return _west_states

    def get_north_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's north
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _north_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpn, slicer=self.NORTH_BOUND_IDX
            )
            for qpn in self.qp.N
        )
        return _north_states

    def get_south_boundary_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's south
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _south_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qps, slicer=self.SOUTH_BOUND_IDX
            )
            for qps in self.qp.S
        )
        return _south_states

    def get_east_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's east
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _east_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpe, slicer=self.EAST_FACE_IDX
            )
            for qpe in self.qp.E
        )
        return _east_states

    def get_west_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's west
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _west_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpw, slicer=self.WEST_FACE_IDX
            )
            for qpw in self.qp.W
        )
        return _west_states

    def get_north_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's north
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _north_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qpn, slicer=self.NORTH_FACE_IDX
            )
            for qpn in self.qp.N
        )
        return _north_states

    def get_south_face_states_at_qp(self) -> tuple[np.ndarray]:
        """
        Return the solution state data in the cells along the block's south
        boundary at each quadrature point.

        :rtype: None
        :return: None
        """
        _south_states = tuple(
            self.fvm.limited_solution_at_quadrature_point(
                state=self.state, refBLK=self, qp=qps, slicer=self.SOUTH_FACE_IDX
            )
            for qps in self.qp.S
        )
        return _south_states


class BlockGeometry:
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
