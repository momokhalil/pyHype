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
from pyHype.utils import utils
from pyHype.mesh.QuadMesh import QuadMesh
from typing import TYPE_CHECKING, Union
from pyHype.mesh import quadratures as qp
from pyHype.states.states import PrimitiveState
from pyHype.blocks.base import BaseBlock_FVM

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput
    from pyHype.blocks.base import QuadBlock, BaseBlock
    from pyHype.mesh.base import BlockDescription


class GhostBlocks:
    def __init__(
        self,
        inputs: ProblemInput,
        block_data: BlockDescription,
        refBLK: BaseBlock,
        state_type: str = "conservative",
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
            inputs, BCtype=block_data.BCTypeE, refBLK=refBLK, state_type=state_type
        )
        self.W = GhostBlockWest(
            inputs, BCtype=block_data.BCTypeW, refBLK=refBLK, state_type=state_type
        )
        self.N = GhostBlockNorth(
            inputs, BCtype=block_data.BCTypeN, refBLK=refBLK, state_type=state_type
        )
        self.S = GhostBlockSouth(
            inputs, BCtype=block_data.BCTypeS, refBLK=refBLK, state_type=state_type
        )

    def __call__(self):
        return self.__dict__.values()

    def clear_cache(self):
        self.E.state.clear_cache()
        self.W.state.clear_cache()
        self.N.state.clear_cache()
        self.S.state.clear_cache()


class BoundaryConditionFunctions:
    def __init__(self):
        pass

    @staticmethod
    def BC_reflection(
        state: np.ndarray, wall_angle: Union[np.ndarray, int, float]
    ) -> None:
        """
        Flips the sign of the u velocity along the wall. Rotates the state from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state)
        state[:, :, 1] = -state[:, :, 1]
        utils.unrotate(wall_angle, state)


class GhostBlock(BaseBlock_FVM, BoundaryConditionFunctions):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: BaseBlock,
        nx: int,
        ny: int,
        state_type: str = "conservative",
    ):

        super().__init__(inputs, nx, ny, state_type=state_type)

        self.BCtype = BCtype
        self.nghost = inputs.nghost
        self.refBLK = refBLK

        self.mesh = None
        self.theta = None

        self.set_BC = None
        # Assign the BCset method to avoid checking type everytime
        if self.BCtype == "None":
            self.set_BC = self.set_BC_none
        elif self.BCtype == "Reflection":
            self.set_BC = self.set_BC_reflection
        elif self.BCtype == "Slipwall":
            self.set_BC = self.set_BC_slipwall
        elif self.BCtype == "InletDirichlet":
            self._inlet_realizability_check()
            self.set_BC = self.set_BC_inlet_dirichlet
        elif self.BCtype == "InletRiemann":
            self._inlet_realizability_check()
            self.set_BC = self.set_BC_inlet_riemann
        elif self.BCtype == "OutletDirichlet":
            self.set_BC = self.set_BC_outlet_dirichlet
        elif self.BCtype == "OutletRiemann":
            self.set_BC = self.set_BC_outlet_riemann
        else:
            raise ValueError(
                "Boundary Condition type "
                + str(self.BCtype)
                + " has not been specialized."
            )

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

    def row(self, index: int, copy: bool = False) -> np.ndarray:
        """
        Return the solution stored in the index-th row of the mesh. For example, if index is 0, then the state at the
        most-bottom row of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which row needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - _row (np.ndarray): The numpy array containing the solution at the index-th row being returned.
        """
        _row = self.state.Q[None, index, :, :]
        return _row.copy() if copy else _row

    def col(self, index: int, copy: bool = False) -> np.ndarray:
        """
        Return the solution stored in the index-th column of the mesh. For example, if index is 0, then the state at the
        left-most column of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which column needs to be returned.
            - copy (bool): To copy the numpy array pr return a view

        Return:
            - (np.ndarray): The numpy array containing the soution at the index-th column being returned.
        """
        _col = self.state.Q[:, None, index, :]
        return _col.copy() if copy else _col

    @abstractmethod
    def set_BC_none(self):
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_outlet_dirichlet(self):
        """
        Set outlet dirichlet boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_outlet_riemann(self):
        """
        Set outlet riemann boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_reflection(self):
        """
        Set reflection boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_slipwall(self):
        """
        Set slipwall boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_inlet_dirichlet(self):
        """
        Set inlet dirichlet boundary condition
        """
        raise NotImplementedError

    @abstractmethod
    def set_BC_inlet_riemann(self):
        """
        Set inlet riemann boundary condition
        """
        raise NotImplementedError


class GhostBlockEast(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: Union[BaseBlock, QuadBlock],
        state_type: str = "conservative",
    ) -> None:

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            nx=inputs.nghost,
            ny=inputs.ny,
            state_type=state_type,
        )

        # Calculate coordinates of all four vertices
        NWx = self.refBLK.mesh.nodes.x[-1, -1]
        NWy = self.refBLK.mesh.nodes.y[-1, -1]
        SWx = self.refBLK.mesh.nodes.x[0, -1]
        SWy = self.refBLK.mesh.nodes.y[0, -1]
        NEx, NEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=self.refBLK.mesh.nodes.x[-1, -1 - self.inputs.nghost],
            yr=self.refBLK.mesh.nodes.y[-1, -1 - self.inputs.nghost],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            SWx,
            SWy,
            xr=self.refBLK.mesh.nodes.x[0, -1 - self.inputs.nghost],
            yr=self.refBLK.mesh.nodes.y[0, -1 - self.inputs.nghost],
        )
        # Construct Mesh
        self.mesh = QuadMesh(
            self.inputs,
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nghost,
            ny=inputs.ny,
        )
        self.QP = qp.QuadraturePointData(inputs, refMESH=self.mesh)

    def set_BC_none(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        if state is None:
            self.state.U = self.refBLK.neighbors.E.get_west_ghost_states()

    def set_BC_reflection(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set reflection boundary condition on the eastern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        _state = self.refBLK.get_east_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.east_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_slipwall(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set slipwall boundary condition on the eastern face, keeps the tangential component as is and zeros the
        normal component.
        """
        _state = self.refBLK.get_east_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.east_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_inlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
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
            self.inputs, W_vector=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if state_type == "conservative":
            state_bc = _W.to_conservative_state()
        elif state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        if state is not None:
            state[:, :, :] = state_bc.Q
        else:
            self.state.U[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        if state is None:
            self.state.U = self.refBLK.get_east_ghost_states()

    def set_BC_inlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError


class GhostBlockWest(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: Union[BaseBlock, QuadBlock],
        state_type: str = "conservative",
    ):

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            nx=inputs.nghost,
            ny=inputs.ny,
            state_type=state_type,
        )

        # Calculate coordinates of all four vertices
        NEx = self.refBLK.mesh.nodes.x[-1, 0]
        NEy = self.refBLK.mesh.nodes.y[-1, 0]
        SEx = self.refBLK.mesh.nodes.x[0, 0]
        SEy = self.refBLK.mesh.nodes.y[0, 0]
        NWx, NWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=self.refBLK.mesh.nodes.x[-1, self.inputs.nghost],
            yr=self.refBLK.mesh.nodes.y[-1, self.inputs.nghost],
        )
        SWx, SWy = utils.reflect_point(
            NEx,
            NEy,
            SEx,
            SEy,
            xr=self.refBLK.mesh.nodes.x[0, self.inputs.nghost],
            yr=self.refBLK.mesh.nodes.y[0, self.inputs.nghost],
        )
        # Construct Mesh
        self.mesh = QuadMesh(
            self.inputs,
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nghost,
            ny=inputs.ny,
        )
        self.QP = qp.QuadraturePointData(inputs, refMESH=self.mesh)

    def set_BC_none(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        if state is None:
            self.state.U = self.refBLK.neighbors.W.get_east_ghost_states()

    def set_BC_reflection(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set reflection boundary condition on the western face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        _state = self.refBLK.get_west_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.west_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_slipwall(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set slipwall boundary condition on the western face, keeps the tangential component as is and zeros the
        normal component.
        """
        _state = self.refBLK.get_west_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.west_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_inlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
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
            self.inputs, W_vector=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if state_type == "conservative":
            state_bc = _W.to_conservative_state()
        elif state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        if state is not None:
            state[:, :, :] = state_bc.Q
        else:
            self.state.U[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        if state is None:
            self.state.U = self.refBLK.get_east_ghost_states()

    def set_BC_inlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError


class GhostBlockNorth(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: Union[BaseBlock, QuadBlock],
        state_type: str = "conservative",
    ):

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            nx=inputs.nx,
            ny=inputs.nghost,
            state_type=state_type,
        )

        # Calculate coordinates of all four vertices
        SWx = self.refBLK.mesh.nodes.x[-1, 0]
        SWy = self.refBLK.mesh.nodes.y[-1, 0]
        SEx = self.refBLK.mesh.nodes.x[-1, -1]
        SEy = self.refBLK.mesh.nodes.y[-1, -1]
        NWx, NWy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, 0],
            yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, 0],
        )
        NEx, NEy = utils.reflect_point(
            SWx,
            SWy,
            SEx,
            SEy,
            xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, -1],
            yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, -1],
        )
        # Construct Mesh
        self.mesh = QuadMesh(
            self.inputs,
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nx,
            ny=inputs.nghost,
        )
        self.QP = qp.QuadraturePointData(inputs, refMESH=self.mesh)

    def set_BC_none(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        if state is None:
            self.state.U = self.refBLK.neighbors.N.get_south_ghost_states()

    def set_BC_reflection(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        _state = self.refBLK.get_north_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.north_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_slipwall(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        _state = self.refBLK.get_north_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.north_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_inlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
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
            self.inputs, W_vector=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if state_type == "conservative":
            state_bc = _W.to_conservative_state()
        elif state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        if state is not None:
            state[:, :, :] = state_bc.Q
        else:
            self.state.U[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        if state is None:
            self.state.U = self.refBLK.get_north_ghost_states()

    def set_BC_inlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError


class GhostBlockSouth(GhostBlock):
    def __init__(
        self,
        inputs: ProblemInput,
        BCtype: str,
        refBLK: Union[BaseBlock, QuadBlock],
        state_type: str = "conservative",
    ) -> None:

        super().__init__(
            inputs,
            BCtype,
            refBLK,
            nx=inputs.nx,
            ny=inputs.nghost,
            state_type=state_type,
        )

        # Calculate coordinates of all four vertices
        NWx = self.refBLK.mesh.nodes.x[0, 0]
        NWy = self.refBLK.mesh.nodes.y[0, 0]
        NEx = self.refBLK.mesh.nodes.x[0, -1]
        NEy = self.refBLK.mesh.nodes.y[0, -1]
        SWx, SWy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, 0],
            yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, 0],
        )
        SEx, SEy = utils.reflect_point(
            NWx,
            NWy,
            NEx,
            NEy,
            xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, -1],
            yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, -1],
        )
        # Construct Mesh
        self.mesh = QuadMesh(
            self.inputs,
            NE=(NEx, NEy),
            NW=(NWx, NWy),
            SE=(SEx, SEy),
            SW=(SWx, SWy),
            nx=inputs.nx,
            ny=inputs.nghost,
        )
        self.QP = qp.QuadraturePointData(inputs, refMESH=self.mesh)

    def set_BC_none(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set no boundary conditions. Equivalent of ensuring two blocks are connected, and allows flow to pass between
        them.
        """
        if state is None:
            self.state.U = self.refBLK.neighbors.S.get_north_ghost_states()

    def set_BC_reflection(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        _state = self.refBLK.get_south_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.south_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_slipwall(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        _state = self.refBLK.get_south_ghost_states() if state is None else state
        self.BC_reflection(_state, self.refBLK.mesh.south_boundary_angle())
        if state is None:
            self.state.U = _state

    def set_BC_inlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
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
            self.inputs, W_vector=np.array([_r, _u, _v, _p]).reshape((1, 1, 4))
        )

        if state_type == "conservative":
            state_bc = _W.to_conservative_state()
        elif state_type == "primitive":
            state_bc = _W
        else:
            raise TypeError(
                "set_BC_inlet_dirichlet() Error! Unknown reconstruction_type"
            )

        if state is not None:
            state[:, :, :] = state_bc.Q
        else:
            self.state.U[:, :, :] = state_bc.Q

    def set_BC_outlet_dirichlet(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ) -> None:
        """
        Set outlet dirichlet boundary condition, by copying values directly adjacent to the boundary into the
        ghost cells.
        """
        if state is None:
            self.state.U = self.refBLK.get_south_ghost_states()

    def set_BC_inlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError

    def set_BC_outlet_riemann(
        self, state: np.ndarray = None, state_type: str = "conservative"
    ):
        return NotImplementedError
