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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

from abc import abstractmethod
from pyhype.utils import utils
from pyhype.mesh.quad_mesh import QuadMesh
from pyhype.mesh import quadratures as quadratures
from typing import TYPE_CHECKING, Type, Callable, Union
from pyhype.boundary_conditions.base import BoundaryCondition
from pyhype.boundary_conditions.mixin import BoundaryConditionMixin
from pyhype.blocks.base import BaseBlockFVM, BlockGeometry, BlockDescription


if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.blocks.base import QuadBlock
    from pyhype.solvers.base import SolverConfig
    from pyhype.mesh.quadratures import QuadraturePointData


class GhostBlocks:
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
        self.E = GhostBlockEast(
            config,
            BCtype=block_data.bc.E,
            parent_block=parent_block,
            state_type=state_type,
        )
        self.W = GhostBlockWest(
            config,
            BCtype=block_data.bc.W,
            parent_block=parent_block,
            state_type=state_type,
        )
        self.N = GhostBlockNorth(
            config,
            BCtype=block_data.bc.N,
            parent_block=parent_block,
            state_type=state_type,
        )
        self.S = GhostBlockSouth(
            config,
            BCtype=block_data.bc.S,
            parent_block=parent_block,
            state_type=state_type,
        )

    def __call__(self):
        return self.__dict__.values()

    def clear_cache(self):
        self.E.state.clear_cache()
        self.W.state.clear_cache()
        self.N.state.clear_cache()
        self.S.state.clear_cache()


class GhostBlock(BaseBlockFVM, BoundaryConditionMixin):
    def __init__(
        self,
        config: SolverConfig,
        BCtype: Union[str, Callable],
        parent_block: QuadBlock,
        state_type: Type[State],
        mesh: QuadMesh = None,
        qp: QuadraturePointData = None,
    ):
        self.theta = None
        self.BCtype = BCtype
        self.parent_block = parent_block
        self.nghost = config.nghost
        self.state_type = state_type

        if self.BCtype is None:
            self._apply_bc_func = self.set_BC_none
        elif self.BCtype == "Reflection":
            self._apply_bc_func = self.set_BC_reflection
        elif self.BCtype == "Slipwall":
            self._apply_bc_func = self.set_BC_slipwall
        elif self.BCtype == "OutletDirichlet":
            self._apply_bc_func = self.set_BC_outlet_dirichlet
        elif self.BCtype == "OutletRiemann":
            self._apply_bc_func = self.set_BC_outlet_riemann
        elif isinstance(self.BCtype, BoundaryCondition):
            self._apply_bc_func = self.BCtype
        else:
            raise ValueError(
                "Boundary Condition type "
                + str(self.BCtype)
                + " has not been specialized."
            )
        super().__init__(config, mesh=mesh, qp=qp, state_type=state_type)

    def __getitem__(self, index):
        return self.state.data[index]

    def realizable(self):
        return self.state.realizable()

    def apply_boundary_condition(self, state: State = None) -> None:
        bc_state = self._get_ghost_state_from_ref_blk() if state is None else state
        self._apply_bc_func(bc_state)
        if state is None:
            self.state = bc_state

    @abstractmethod
    def _get_ghost_state_from_ref_blk(self):
        raise NotImplementedError

    @abstractmethod
    def set_BC_none(self, state: State):
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_outlet_dirichlet(self, state: State):
        """
        Set outlet dirichlet boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_outlet_riemann(self, state: State):
        """
        Set outlet riemann boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_reflection(self, state: State):
        """
        Set reflection boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_slipwall(self, state: State):
        """
        Set slipwall boundary condition
        """
        raise NotImplementedError


class GhostBlockEast(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        BCtype: str,
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
            config,
            BCtype,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.parent_block.get_east_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.from_state(self.parent_block.neighbors.E.get_west_ghost_states())

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the eastern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.east_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the eastern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.east_boundary_angle())

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockWest(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        BCtype: str,
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
            BCtype,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.parent_block.get_west_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.from_state(self.parent_block.neighbors.W.get_east_ghost_states())

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the western face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.west_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State = None,
    ) -> None:
        """
        Set slipwall boundary condition on the western face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.west_boundary_angle())

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockNorth(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        BCtype: str,
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
            BCtype,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.parent_block.get_north_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.from_state(self.parent_block.neighbors.N.get_south_ghost_states())

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.north_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.north_boundary_angle())

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockSouth(GhostBlock):
    def __init__(
        self,
        config: SolverConfig,
        BCtype: str,
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
            BCtype,
            parent_block,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.parent_block.get_south_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.from_state(self.parent_block.neighbors.S.get_north_ghost_states())

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.south_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.parent_block.mesh.south_boundary_angle())

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError
