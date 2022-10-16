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

import numpy as np
from abc import abstractmethod
from pyhype.utils import utils
from types import FunctionType
from pyhype.mesh.quad_mesh import QuadMesh
from typing import TYPE_CHECKING, Union, Type
from pyhype.mesh import quadratures as quadratures
from pyhype.states.primitive import PrimitiveState
from pyhype.states.conservative import ConservativeState
from pyhype.blocks.base import BaseBlockFVM, BlockGeometry, BlockDescription
from pyhype.blocks.base import BlockMixin

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.solvers.base import ProblemInput
    from pyhype.blocks.base import QuadBlock, BaseBlock
    from pyhype.mesh.quadratures import QuadraturePointData


class GhostBlocks:
    def __init__(
        self,
        inputs: ProblemInput,
        block_data: BlockDescription,
        refBLK: QuadBlock,
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
            inputs,
            BCtype=block_data.bc.E,
            refBLK=refBLK,
            state_type=state_type,
        )
        self.W = GhostBlockWest(
            inputs,
            BCtype=block_data.bc.W,
            refBLK=refBLK,
            state_type=state_type,
        )
        self.N = GhostBlockNorth(
            inputs,
            BCtype=block_data.bc.N,
            refBLK=refBLK,
            state_type=state_type,
        )
        self.S = GhostBlockSouth(
            inputs,
            BCtype=block_data.bc.S,
            refBLK=refBLK,
            state_type=state_type,
        )

    def __call__(self):
        return self.__dict__.values()

    def clear_cache(self):
        self.E.state.clear_cache()
        self.W.state.clear_cache()
        self.N.state.clear_cache()
        self.S.state.clear_cache()


class BoundaryConditionMixin:
    @staticmethod
    def BC_reflection(state: State, wall_angle: Union[np.ndarray, int, float]) -> None:
        """
        Flips the sign of the u velocity along the wall. Rotates the state from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state.Q)
        state.Q[:, :, 1] = -state.Q[:, :, 1]
        utils.unrotate(wall_angle, state.Q)


class GhostBlock(BaseBlockFVM, BoundaryConditionMixin, BlockMixin):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: QuadBlock,
        state_type: Type[State],
        mesh: QuadMesh = None,
        qp: QuadraturePointData = None,
    ):
        self.theta = None
        self.BCtype = BCtype
        self.refBLK = refBLK
        self.nghost = inputs.nghost
        self.state_type = state_type

        if self.BCtype is None:
            self._apply_bc_func = self.set_BC_none
        elif self.BCtype == "Reflection":
            self._apply_bc_func = self.set_BC_reflection
        elif self.BCtype == "Slipwall":
            self._apply_bc_func = self.set_BC_slipwall
        elif self.BCtype == "InletDirichlet":
            self._inlet_realizability_check()
            self._apply_bc_func = self.set_BC_inlet_dirichlet
        elif self.BCtype == "InletRiemann":
            self._inlet_realizability_check()
            self._apply_bc_func = self.set_BC_inlet_riemann
        elif self.BCtype == "OutletDirichlet":
            self._apply_bc_func = self.set_BC_outlet_dirichlet
        elif self.BCtype == "OutletRiemann":
            self._apply_bc_func = self.set_BC_outlet_riemann
        elif isinstance(self.BCtype, FunctionType):
            self._apply_bc_func = self.BCtype
        else:
            raise ValueError(
                "Boundary Condition type "
                + str(self.BCtype)
                + " has not been specialized."
            )
        super().__init__(inputs, mesh=mesh, qp=qp, state_type=state_type)

    def _inlet_realizability_check(self):
        """
        Realizability check for inlet conditions. Ensures positive density and pressure/energy. Raises ValueError if
        not realizable and TypeError if not float or int.

        Parameters:
            - N/A

        Returns:
            - N/A
        """

        _inlets = [
            "BC_inlet_west_rho",
            "BC_inlet_east_rho",
            "BC_inlet_north_rho",
            "BC_inlet_south_rho",
            "BC_inlet_west_p",
            "BC_inlet_east_p",
            "BC_inlet_north_p",
            "BC_inlet_south_p",
        ]

        _has_inlets = [inlet for inlet in _inlets if hasattr(self.inputs, inlet)]

        for ic in _has_inlets:
            if self.inputs.__getattribute__(ic) <= 0:
                raise ValueError(
                    "Inlet density or pressure is less than or equal to zero and thus non-physical."
                )
            if not isinstance(self.inputs.__getattribute__(ic), (int, float)):
                raise TypeError(
                    "Inlet density or pressure is not of type int or float."
                )

    def __getitem__(self, index):
        return self.state.U[index]

    def realizable(self):
        return self.state.realizable()

    def apply_boundary_condition(self, state: State = None) -> None:
        _bc_state = self._get_ghost_state_from_ref_blk() if state is None else state
        self._apply_bc_func(_bc_state)
        if state is None:
            self.state = _bc_state

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

    @abstractmethod
    def set_BC_inlet_dirichlet(self, state: State):
        """
        Set inlet dirichlet boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_inlet_riemann(self, state: State):
        """
        Set inlet riemann boundary condition
        """
        raise NotImplementedError


class GhostBlockEast(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: QuadBlock,
        state_type: Type[State],
    ) -> None:

        # Calculate coordinates of all four vertices
        NWx = refBLK.mesh.nodes.x[-1, -1]
        NWy = refBLK.mesh.nodes.y[-1, -1]
        SWx = refBLK.mesh.nodes.x[0, -1]
        SWy = refBLK.mesh.nodes.y[0, -1]
        NEx, NEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=refBLK.mesh.nodes.x[-1, -1 - inputs.nghost],
            yr=refBLK.mesh.nodes.y[-1, -1 - inputs.nghost],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=refBLK.mesh.nodes.x[0, -1 - inputs.nghost],
            yr=refBLK.mesh.nodes.y[0, -1 - inputs.nghost],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nghost,
            ny=inputs.ny,
        )
        mesh = QuadMesh(inputs, block_geometry=block_geometry)
        qp = quadratures.QuadraturePointData(inputs, refMESH=mesh)

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.refBLK.get_east_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.Q = self.refBLK.neighbors.E.get_west_ghost_states()

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the eastern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.east_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the eastern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.east_boundary_angle())

    def set_BC_inlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set dirichlet inlet boundary condition on the eastern face.
        """
        _g = self.inputs.gamma - 1
        _r = self.inputs.BC_inlet_east_rho
        _u = self.inputs.BC_inlet_east_u
        _v = self.inputs.BC_inlet_east_v
        _p = self.inputs.BC_inlet_east_p

        _W = PrimitiveState(
            self.inputs, array=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if self.state_type == "conservative":
            state_bc = ConservativeState(self.inputs, state=_W)
        elif self.state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )
        state.Q[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_inlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockWest(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: QuadBlock,
        state_type: Type[State],
    ):
        # Calculate coordinates of all four vertices
        NEx = refBLK.mesh.nodes.x[-1, 0]
        NEy = refBLK.mesh.nodes.y[-1, 0]
        SEx = refBLK.mesh.nodes.x[0, 0]
        SEy = refBLK.mesh.nodes.y[0, 0]
        NWx, NWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=refBLK.mesh.nodes.x[-1, inputs.nghost],
            yr=refBLK.mesh.nodes.y[-1, inputs.nghost],
        )
        SWx, SWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=refBLK.mesh.nodes.x[0, inputs.nghost],
            yr=refBLK.mesh.nodes.y[0, inputs.nghost],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nghost,
            ny=inputs.ny,
        )
        mesh = QuadMesh(
            inputs,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(inputs, refMESH=mesh)

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.refBLK.get_west_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.Q = self.refBLK.neighbors.W.get_east_ghost_states()

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the western face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.west_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State = None,
    ) -> None:
        """
        Set slipwall boundary condition on the western face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.west_boundary_angle())

    def set_BC_inlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set dirichlet inlet boundary condition on the eastern face.
        """
        _g = self.inputs.gamma - 1
        _r = self.inputs.BC_inlet_west_rho
        _u = self.inputs.BC_inlet_west_u
        _v = self.inputs.BC_inlet_west_v
        _p = self.inputs.BC_inlet_west_p

        _W = PrimitiveState(
            self.inputs, array=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if self.state_type == "conservative":
            state_bc = ConservativeState(self.inputs, state=_W)
        elif self.state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        state.Q[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_inlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockNorth(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: QuadBlock,
        state_type: Type[State],
    ):
        # Calculate coordinates of all four vertices
        SWx = refBLK.mesh.nodes.x[-1, 0]
        SWy = refBLK.mesh.nodes.y[-1, 0]
        SEx = refBLK.mesh.nodes.x[-1, -1]
        SEy = refBLK.mesh.nodes.y[-1, -1]
        NWx, NWy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=refBLK.mesh.nodes.x[-1 - inputs.nghost, 0],
            yr=refBLK.mesh.nodes.y[-1 - inputs.nghost, 0],
        )
        NEx, NEy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=refBLK.mesh.nodes.x[-1 - inputs.nghost, -1],
            yr=refBLK.mesh.nodes.y[-1 - inputs.nghost, -1],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nx,
            ny=inputs.nghost,
        )
        mesh = QuadMesh(
            inputs,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(inputs, refMESH=mesh)

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.refBLK.get_north_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.Q = self.refBLK.neighbors.N.get_south_ghost_states()

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.north_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.north_boundary_angle())

    def set_BC_inlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set dirichlet inlet boundary condition on the northern face.
        """
        _g = self.inputs.gamma - 1
        _r = self.inputs.BC_inlet_north_rho
        _u = self.inputs.BC_inlet_north_u
        _v = self.inputs.BC_inlet_north_v
        _p = self.inputs.BC_inlet_north_p

        _W = PrimitiveState(
            self.inputs, array=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if self.state_type == "conservative":
            state_bc = ConservativeState(self.inputs, state=_W)
        elif self.state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        state.Q[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_inlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError


class GhostBlockSouth(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: QuadBlock,
        state_type: Type[State],
    ) -> None:
        # Calculate coordinates of all four vertices
        NWx = refBLK.mesh.nodes.x[0, 0]
        NWy = refBLK.mesh.nodes.y[0, 0]
        NEx = refBLK.mesh.nodes.x[0, -1]
        NEy = refBLK.mesh.nodes.y[0, -1]
        SWx, SWy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=refBLK.mesh.nodes.x[inputs.nghost, 0],
            yr=refBLK.mesh.nodes.y[inputs.nghost, 0],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=refBLK.mesh.nodes.x[inputs.nghost, -1],
            yr=refBLK.mesh.nodes.y[inputs.nghost, -1],
        )
        # Construct Mesh
        block_geometry = BlockGeometry(
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nx,
            ny=inputs.nghost,
        )
        mesh = QuadMesh(
            inputs,
            block_geometry=block_geometry,
        )
        qp = quadratures.QuadraturePointData(inputs, refMESH=mesh)

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            mesh=mesh,
            qp=qp,
            state_type=state_type,
        )

    def _get_ghost_state_from_ref_blk(self):
        return self.refBLK.get_south_ghost_states()

    def set_BC_none(
        self,
        state: State,
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        state.Q = self.refBLK.neighbors.S.get_north_ghost_states()

    def set_BC_reflection(
        self,
        state: State,
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.south_boundary_angle())

    def set_BC_slipwall(
        self,
        state: State,
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        self.BC_reflection(state, self.refBLK.mesh.south_boundary_angle())

    def set_BC_inlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set dirichlet inlet boundary condition on the southern face.
        """
        _g = self.inputs.gamma - 1
        _r = self.inputs.BC_inlet_south_rho
        _u = self.inputs.BC_inlet_south_u
        _v = self.inputs.BC_inlet_south_v
        _p = self.inputs.BC_inlet_south_p

        _W = PrimitiveState(
            self.inputs, array=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if self.state_type == "conservative":
            state_bc = ConservativeState(self.inputs, state=_W)
        elif self.state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        state[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self,
        state: State,
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        pass

    def set_BC_inlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self,
        state: State,
    ):
        return NotImplementedError
