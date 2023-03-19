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
import logging
from typing import TYPE_CHECKING, Type, Callable, Union, Optional

import mpi4py.MPI
import numpy as np

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

from mpi4py import MPI
from pyhype.utils import utils
from pyhype.mesh.quad_mesh import QuadMesh
from pyhype.mesh import quadratures as quadratures
from pyhype.boundary_conditions.base import BoundaryCondition
from pyhype.utils.utils import NumpySlice, SidePropertyDict, Direction
from pyhype.boundary_conditions.funcs import BoundaryConditionFunctions
from pyhype.blocks.base import BaseBlockFVM, BlockGeometry, BlockDescription


if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.solvers.base import SolverConfig
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.mesh.quadratures import QuadraturePointData


class GhostBlocks(SidePropertyDict):
    def __init__(
        self,
        config: SolverConfig,
        block_data: BlockDescription,
        parent_block: QuadBlock,
        state_type: Type[State],
    ) -> None:
        """
        A class designed to hold references to each Block's ghost blocks.

        Parameters:
            - E: Reference to the east ghost block
            - W: Reference to the west nghost block
            - N: Reference to the north ghost block
            - S: Reference to the south ghost block
        """
        self.cpu = MPI.COMM_WORLD.Get_rank()

        E = GhostBlockEast(
            config,
            bc_type=block_data.bc.E,
            parent_block=parent_block,
            state_type=state_type,
        )
        W = GhostBlockWest(
            config,
            bc_type=block_data.bc.W,
            parent_block=parent_block,
            state_type=state_type,
        )
        N = GhostBlockNorth(
            config,
            bc_type=block_data.bc.N,
            parent_block=parent_block,
            state_type=state_type,
        )
        S = GhostBlockSouth(
            config,
            bc_type=block_data.bc.S,
            parent_block=parent_block,
            state_type=state_type,
        )
        super().__init__(E=E, W=W, N=N, S=S)

    def get_blocks(self):
        return self.values()

    def clear_cache(self):
        self.E.state.clear_cache()
        self.W.state.clear_cache()
        self.N.state.clear_cache()
        self.S.state.clear_cache()


class GhostBlock(BaseBlockFVM):
    def __init__(
        self,
        config: SolverConfig,
        bc_type: Union[str, Callable],
        parent_block: QuadBlock,
        state_type: Type[State],
        direction: int,
        mesh: QuadMesh = None,
        qp: QuadraturePointData = None,
    ):
        self.dir = direction
        self.theta = None
        self.bc_type = bc_type
        self.parent_block = parent_block
        self.nghost = config.nghost
        self.state_type = state_type

        self.mpi = MPI.COMM_WORLD
        self.cpu = self.mpi.Get_rank()
        self.shape = (mesh.shape[0], mesh.shape[1], 4)

        self._num_elements = mesh.shape[0] * mesh.shape[1] * 4
        self._recv_buf = np.zeros((self._num_elements,), dtype=np.float64)
        self._ghost_idx = NumpySlice.ghost(nghost=config.nghost)
        self._logger = logging.getLogger(self.__class__.__name__)

        if self.bc_type is None:
            self._apply_bc_func = self._apply_none_bc
        elif self.bc_type == "Reflection":
            self._apply_bc_func = self._apply_reflection_bc
        elif self.bc_type == "Slipwall":
            self._apply_bc_func = self._apply_slipwall_bc
        elif self.bc_type == "OutletDirichlet":
            self._apply_bc_func = self._apply_outlet_dirichlet_bc
        elif isinstance(self.bc_type, BoundaryCondition):
            self._apply_bc_func = self.bc_type
        else:
            raise ValueError(
                "Boundary Condition type "
                + str(self.bc_type)
                + " has not been specialized."
            )
        super().__init__(config, mesh=mesh, qp=qp, state_type=state_type)

    def __getitem__(self, index):
        return self.state.data[index]

    def realizable(self) -> bool:
        """
        Runs a physical realizability check on the block's State.

        :return: bool representing the State's realizability
        """
        return self.state.realizable()

    def apply_boundary_condition(self) -> None:
        """
        Applies the boundary condition to the block's State. First, the State is
        filled with the appropriate state data from either the parent block or the
        parent block's neighbor on this ghost block's side, then the bc function is
        applied to this filled state.

        :return: None
        """
        # self._fill()
        self._apply_bc_func(self.state)

    def apply_boundary_condition_to_state(self, state: State) -> None:
        """
        Applies the boundary condition function to the given State.

        :param state: State object to apply the bc to
        :return: None
        """
        self._apply_bc_func(state)

    def _send_mpi_buffer(self):
        neighbor_blk, neighbor_cpu = self.parent_block.neighbors[self.dir]
        send_buf = self.parent_block.state[self._ghost_idx[self.dir]].data
        send_req = self.mpi.Isend(
            buf=send_buf.ravel(),
            dest=neighbor_cpu,
            tag=utils.cantor_pair(self.parent_block.global_nBLK, neighbor_blk),
        )
        return send_req

    def send_boundary_data(self) -> [Optional[mpi4py.MPI.Request]]:
        if self.bc_type is not None or self.parent_block.neighbors[self.dir] is None:
            self.state.from_state(self.parent_block.state[self._ghost_idx[self.dir]])
            return
        if isinstance(self.parent_block.neighbors[self.dir], tuple):
            return self._send_mpi_buffer()
        self.state.from_state(
            self.parent_block.neighbors[self.dir].state[self._ghost_idx[-self.dir]]
        )

    def recieve_boundary_data(self):
        if isinstance(self.parent_block.neighbors[self.dir], tuple):
            neighbor_blk, neighbor_cpu = self.parent_block.neighbors[self.dir]
            tag = utils.cantor_pair(neighbor_blk, self.parent_block.global_nBLK)
            req = self.mpi.Irecv(self._recv_buf, source=neighbor_cpu, tag=tag)
            return req

    def apply_recv_buffers_to_state(self):
        if isinstance(self.parent_block.neighbors[self.dir], tuple):
            self.state.data = self._recv_buf.copy().reshape(self.shape)

    def _fill(self) -> None:
        """
        Fills this ghost block's State using the parent block's State data on the
        appropriate side, or the parent blocks neighbor on the appropriate side.

        :return: None
        """
        if self.bc_type is None:
            neighbor = self.parent_block.neighbors[self.dir]
            if neighbor is not None and not isinstance(neighbor, tuple):
                self.state.from_state(
                    self.parent_block.neighbors[self.dir].state[
                        self._ghost_idx[-self.dir]
                    ]
                )
            else:
                neighbor_blk, neighbor_cpu = self.parent_block.neighbors[self.dir]
                self._send_buf[:] = self.parent_block.state[
                    self._ghost_idx[self.dir]
                ].data.ravel()
                self.mpi.Send(
                    self._send_buf, dest=neighbor_cpu, tag=self.parent_block.global_nBLK
                )
                self.mpi.Recv(self._recv_buf, source=neighbor_cpu, tag=neighbor_blk)
                self.state.data = self._recv_buf.reshape(self.shape)
        else:
            self.state.from_state(self.parent_block.state[self._ghost_idx[self.dir]])

    def _apply_none_bc(self, state: State) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected,
        and allows flow to pass between them.
        """
        pass

    def _apply_outlet_dirichlet_bc(self, state: State) -> None:
        """
        Set outlet dirichlet boundary condition
        """
        pass

    def _apply_reflection_bc(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential
        component as is and reverses the sign of the normal component.
        """
        BoundaryConditionFunctions.reflection(
            state, self.parent_block.mesh.boundary_angle(direction=self.dir)
        )

    def _apply_slipwall_bc(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        BoundaryConditionFunctions.reflection(
            state, self.parent_block.mesh.boundary_angle(direction=self.dir)
        )


class GhostBlockEast(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        bc_type: str,
        parent_block: QuadBlock,
        state_type: Type[State],
    ) -> None:

        # Calculate coordinates of all four vertices
        NWx = parent_block.mesh.nodes.x[-1, -1]
        NWy = parent_block.mesh.nodes.y[-1, -1]
        SWx = parent_block.mesh.nodes.x[0, -1]
        SWy = parent_block.mesh.nodes.y[0, -1]
        NEx, NEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=parent_block.mesh.nodes.x[-1, -1 - config.nghost],
            yr=parent_block.mesh.nodes.y[-1, -1 - config.nghost],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=parent_block.mesh.nodes.x[0, -1 - config.nghost],
            yr=parent_block.mesh.nodes.y[0, -1 - config.nghost],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=config.nghost,
            ny=config.ny,
        )
        mesh = QuadMesh(config, block_geometry=block_geometry)
        qp = quadratures.QuadraturePointData(config, refMESH=mesh)

        super().__init__(
            config=config,
            bc_type=bc_type,
            parent_block=parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
            direction=Direction.east,
        )


class GhostBlockWest(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        bc_type: str,
        parent_block: QuadBlock,
        state_type: Type[State],
    ):
        # Calculate coordinates of all four vertices
        NEx = parent_block.mesh.nodes.x[-1, 0]
        NEy = parent_block.mesh.nodes.y[-1, 0]
        SEx = parent_block.mesh.nodes.x[0, 0]
        SEy = parent_block.mesh.nodes.y[0, 0]
        NWx, NWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=parent_block.mesh.nodes.x[-1, config.nghost],
            yr=parent_block.mesh.nodes.y[-1, config.nghost],
        )
        SWx, SWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=parent_block.mesh.nodes.x[0, config.nghost],
            yr=parent_block.mesh.nodes.y[0, config.nghost],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=config.nghost,
            ny=config.ny,
        )
        mesh = QuadMesh(
            config,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(config, refMESH=mesh)

        super().__init__(
            config,
            bc_type,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
            direction=Direction.west,
        )


class GhostBlockNorth(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        bc_type: str,
        parent_block: QuadBlock,
        state_type: Type[State],
    ):
        # Calculate coordinates of all four vertices
        SWx = parent_block.mesh.nodes.x[-1, 0]
        SWy = parent_block.mesh.nodes.y[-1, 0]
        SEx = parent_block.mesh.nodes.x[-1, -1]
        SEy = parent_block.mesh.nodes.y[-1, -1]
        NWx, NWy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=parent_block.mesh.nodes.x[-1 - config.nghost, 0],
            yr=parent_block.mesh.nodes.y[-1 - config.nghost, 0],
        )
        NEx, NEy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=parent_block.mesh.nodes.x[-1 - config.nghost, -1],
            yr=parent_block.mesh.nodes.y[-1 - config.nghost, -1],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=config.nx,
            ny=config.nghost,
        )
        mesh = QuadMesh(
            config,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(config, refMESH=mesh)

        super().__init__(
            config,
            bc_type,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
            direction=Direction.north,
        )


class GhostBlockSouth(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        bc_type: str,
        parent_block: QuadBlock,
        state_type: Type[State],
    ) -> None:
        # Calculate coordinates of all four vertices
        NWx = parent_block.mesh.nodes.x[0, 0]
        NWy = parent_block.mesh.nodes.y[0, 0]
        NEx = parent_block.mesh.nodes.x[0, -1]
        NEy = parent_block.mesh.nodes.y[0, -1]
        SWx, SWy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=parent_block.mesh.nodes.x[config.nghost, 0],
            yr=parent_block.mesh.nodes.y[config.nghost, 0],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=parent_block.mesh.nodes.x[config.nghost, -1],
            yr=parent_block.mesh.nodes.y[config.nghost, -1],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=config.nx,
            ny=config.nghost,
        )
        mesh = QuadMesh(
            config,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(config, refMESH=mesh)

        super().__init__(
            config,
            bc_type,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
            direction=Direction.south,
        )
