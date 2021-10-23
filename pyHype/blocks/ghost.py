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
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from abc import abstractmethod
from pyHype.utils import utils
from pyHype.mesh.QuadMesh import QuadMesh
from typing import TYPE_CHECKING, Union
from pyHype.states import ConservativeState

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput
    from pyHype.blocks.base import QuadBlock

_DEFINED_BC_ = ['None',
                'Reflection',
                'Slipwall',
                'InletDirichlet',
                'InletRiemann',
                'OutletDirichlet',
                'OuletRiemann'
                ]


def is_defined_BC(name: str):
    return True if name in _DEFINED_BC_ else False

class GhostBlock:
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        self.inputs = inputs
        self.BCtype = BCtype
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.nghost = inputs.nghost
        self.refBLK = refBLK

        self.state = None
        self.mesh = None
        self.theta = None

        # Assign the BCset method to avoid checking type everytime
        if self.BCtype in _DEFINED_BC_:
            if self.BCtype == 'None':
                self.set_BC = self.set_BC_none
            elif self.BCtype == 'Reflection':
                self.set_BC = self.set_BC_reflection
            elif self.BCtype == 'Slipwall':
                self.set_BC = self.set_BC_slipwall
            elif self.BCtype == 'InletDirichlet':
                self._inlet_realizability_check()
                self.set_BC = self.set_BC_inlet_dirichlet
            elif self.BCtype == 'InletRiemann':
                self._inlet_realizability_check()
                self.set_BC = self.set_BC_inlet_riemann
            elif self.BCtype == 'OutletDirichlet':
                self.set_BC = self.set_BC_outlet_dirichlet
            elif self.BCtype == 'OutletRiemann':
                self.set_BC = self.set_BC_outlet_riemann
        else:
            raise ValueError('Boundary Condition type ' + str(self.BCtype) + ' has not been specialized.')

    def _inlet_realizability_check(self):
        """
        Realizability check for inlet conditions. Ensures positive density and pressure/energy. Raises ValueError if
        not realizable and TypeError if not float or int.

        Parameters:
            - N/A

        Returns:
            - N/A
        """

        _inlets = ['BC_inlet_west_rho', 'BC_inlet_east_rho', 'BC_inlet_north_rho', 'BC_inlet_south_rho',
                   'BC_inlet_west_p', 'BC_inlet_east_p', 'BC_inlet_north_p', 'BC_inlet_south_p']

        _has_inlets = [inlet for inlet in _inlets if hasattr(self.inputs, inlet)]

        for ic in _has_inlets:
            if self.inputs.__getattribute__(ic) <= 0:
                raise ValueError('Inlet density or pressure is less than or equal to zero and thus non-physical.')
            elif not isinstance(self.inputs.__getattribute__(ic), (int, float)):
                raise TypeError('Inlet density or pressure is not of type int or float.')


    def __getitem__(self, index):
        return self.state.U[index]

    def realizable(self):
        return self.state.realizable()

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
            - _row (np.ndarray): The numpy array containing the solution at the index-th row being returned.
        """
        _row = self.state.Q[None, index, :, :]
        return _row.copy() if copy else _row

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
        _col = self.state.Q[:, None, index, :]
        return _col.copy() if copy else _col

    @staticmethod
    def zero_wall_u_velocity(state: np.ndarray, wall_angle: Union[np.ndarray, int, float]) -> None:
        """
        Sets the u velocity along the wall to zero. Rotates the state from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state)
        state[:, :, 1] = 0
        utils.unrotate(wall_angle, state)

    @staticmethod
    def flip_wall_u_velocity(state: np.ndarray, wall_angle: Union[np.ndarray, int, float]) -> None:
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

    @staticmethod
    def zero_wall_v_velocity(state: np.ndarray, wall_angle: Union[np.ndarray, int, float]) -> None:
        """
        Sets the v velocity along the wall to zero. Rotates the state from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state)
        state[:, :, 2] = 0
        utils.unrotate(wall_angle, state)

    @staticmethod
    def flip_wall_v_velocity(state: np.ndarray, wall_angle: Union[np.ndarray, int, float]) -> None:
        """
        Flips the sign of the v velocity along the wall. Rotates the state from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state)
        state[:, :, 2] = -state[:, :, 2]
        utils.unrotate(wall_angle, state)

    @abstractmethod
    def set_BC_none(self):
        pass

    @abstractmethod
    def set_BC_outlet_dirichlet(self):
        pass

    @abstractmethod
    def set_BC_outlet_riemann(self):
        pass

    @abstractmethod
    def set_BC_reflection(self):
        pass

    @abstractmethod
    def set_BC_slipwall(self):
        pass

    @abstractmethod
    def set_BC_inlet_dirichlet(self):
        pass

    @abstractmethod
    def set_BC_inlet_riemann(self):
        pass


class GhostBlockEast(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        super().__init__(inputs, BCtype, refBLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        # Calculate location of NorthWest corner
        NWx = self.refBLK.mesh.nodes.x[-1, -1]
        NWy = self.refBLK.mesh.nodes.y[-1, -1]
        # Calculate location of SouthWest corner
        SWx = self.refBLK.mesh.nodes.x[0, -1]
        SWy = self.refBLK.mesh.nodes.y[0, -1]

        # Get points on the outside of the block
        NEx, NEy = utils.reflect_point(NWx, NWy, SWx, SWy, xr=self.refBLK.mesh.nodes.x[-1, -1 - self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[-1, -1 - self.inputs.nghost])
        SEx, SEy = utils.reflect_point(NWx, NWy, SWx, SWy, xr=self.refBLK.mesh.nodes.x[0, -1 - self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[0, -1 - self.inputs.nghost])
        # Construct Mesh
        self.mesh = QuadMesh(self.inputs, NE=(NEx, NEy), NW=(NWx, NWy), SE=(SEx, SEy), SW=(SWx, SWy),
                             nx=inputs.nghost, ny=self.refBLK.ny)

    def set_BC_none(self):
        """
        Boundary condition type NONE, allows flow to pass between blocks.
        """
        self.state.U = self.refBLK.neighbors.E.get_west_ghost()

    def set_BC_reflection(self):
        """
        Set reflection boundary condition on the eastern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        state = self.refBLK.get_east_ghost()
        self.flip_wall_u_velocity(state, self.refBLK.mesh.get_east_face_angle())
        self.state.U = state

    def set_BC_slipwall(self):
        """
        Set slipwall boundary condition on the eastern face, keeps the tangential component as is and zeros the
        normal component.
        """
        state = self.refBLK.get_east_ghost()
        self.zero_wall_u_velocity(state, self.refBLK.mesh.get_east_face_angle())
        self.state.U = state

    def set_BC_inlet_dirichlet(self):
        """
        Set dirichlet inlet boundary condition on the eastern face.
        """
        self.state.U[:, :, self.state.RHO_IDX]  = self.inputs.BC_inlet_east_rho
        self.state.U[:, :, self.state.RHOU_IDX] = self.inputs.BC_inlet_east_rho * self.inputs.BC_inlet_east_u
        self.state.U[:, :, self.state.RHOV_IDX] = self.inputs.BC_inlet_east_rho * self.inputs.BC_inlet_east_v
        self.state.U[:, :, self.state.E_IDX]    = self.inputs.BC_inlet_east_p / (self.inputs.gamma - 1) \
                                                + 0.5 * self.inputs.BC_inlet_east_rho \
                                                      * (self.inputs.BC_inlet_east_u ** 2 +
                                                         self.inputs.BC_inlet_east_v ** 2)

    def set_BC_outlet_dirichlet(self):
        self.state.U = self.refBLK.get_east_ghost()

    def set_BC_inlet_riemann(self):
        pass

    def set_BC_outlet_riemann(self):
        pass


class GhostBlockWest(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        super().__init__(inputs, BCtype, refBLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        # Calculate location of NorthEast corner
        NEx = self.refBLK.mesh.nodes.x[-1, 0]
        NEy = self.refBLK.mesh.nodes.y[-1, 0]
        # Calculate location of SouthEast corner
        SEx = self.refBLK.mesh.nodes.x[0, 0]
        SEy = self.refBLK.mesh.nodes.y[0, 0]

        # Get points on the outside of the block
        NWx, NWy = utils.reflect_point(NEx, NEy, SEx, SEy, xr=self.refBLK.mesh.nodes.x[-1, self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[-1, self.inputs.nghost])
        SWx, SWy = utils.reflect_point(NEx, NEy, SEx, SEy, xr=self.refBLK.mesh.nodes.x[0, self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[0, self.inputs.nghost])
        # Construct Mesh
        self.mesh = QuadMesh(self.inputs, NE=(NEx, NEy), NW=(NWx, NWy), SE=(SEx, SEy), SW=(SWx, SWy),
                             nx=inputs.nghost, ny=self.refBLK.ny)

    def set_BC_none(self):
        self.state.U = self.refBLK.neighbors.W.get_east_ghost()

    def set_BC_reflection(self):
        """
        Set reflection boundary condition on the western face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        state = self.refBLK.get_west_ghost()
        self.flip_wall_u_velocity(state, self.refBLK.mesh.get_west_face_angle())
        self.state.U = state

    def set_BC_slipwall(self):
        """
        Set slipwall boundary condition on the western face, keeps the tangential component as is and zeros the
        normal component.
        """
        state = self.refBLK.get_west_ghost()
        self.zero_wall_u_velocity(state, self.refBLK.mesh.get_west_face_angle())
        self.state.U = state

    def set_BC_inlet_dirichlet(self):
        """
        Set dirichlet inlet boundary condition on the eastern face.
        """
        self.state.U[:, :, self.state.RHO_IDX]  = self.inputs.BC_inlet_west_rho
        self.state.U[:, :, self.state.RHOU_IDX] = self.inputs.BC_inlet_west_rho * self.inputs.BC_inlet_west_u
        self.state.U[:, :, self.state.RHOV_IDX] = self.inputs.BC_inlet_west_rho * self.inputs.BC_inlet_west_v
        self.state.U[:, :, self.state.E_IDX]    = self.inputs.BC_inlet_west_p / (self.inputs.gamma - 1) \
                                                + 0.5 * self.inputs.BC_inlet_west_rho \
                                                      * (self.inputs.BC_inlet_west_u ** 2 +
                                                         self.inputs.BC_inlet_west_v ** 2)

    def set_BC_outlet_dirichlet(self):
        self.state.U = self.refBLK.get_west_ghost()

    def set_BC_inlet_riemann(self):
        pass

    def set_BC_outlet_riemann(self):
        pass


class GhostBlockNorth(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        super().__init__(inputs, BCtype, refBLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        # Calculate location of SouthWest corner
        SWx = self.refBLK.mesh.nodes.x[-1, 0]
        SWy = self.refBLK.mesh.nodes.y[-1, 0]
        # Calculate location of SouthEast corner
        SEx = self.refBLK.mesh.nodes.x[-1, -1]
        SEy = self.refBLK.mesh.nodes.y[-1, -1]

        # Get points on the outside of the block
        NWx, NWy = utils.reflect_point(SWx, SWy, SEx, SEy, xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, 0],
                                       yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, 0])
        NEx, NEy = utils.reflect_point(SWx, SWy, SEx, SEy, xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, -1],
                                       yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, -1])
        # Construct Mesh
        self.mesh = QuadMesh(self.inputs, NE=(NEx, NEy), NW=(NWx, NWy), SE=(SEx, SEy), SW=(SWx, SWy),
                             nx=self.refBLK.ny, ny=inputs.nghost)

    def set_BC_none(self):
        self.state.U = self.refBLK.neighbors.N.get_south_ghost()

    def set_BC_reflection(self):
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        state = self.refBLK.get_north_ghost()
        self.flip_wall_v_velocity(state, self.refBLK.mesh.get_north_face_angle() - np.pi / 2)
        self.state.U = state

    def set_BC_slipwall(self):
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        state = self.refBLK.get_north_ghost()
        self.zero_wall_v_velocity(state, self.refBLK.mesh.get_north_face_angle() - np.pi / 2)
        self.state.U = state

    def set_BC_inlet_dirichlet(self):
        """
        Set dirichlet inlet boundary condition on the northern face.
        """
        self.state.U[:, :, self.state.RHO_IDX]  = self.inputs.BC_inlet_north_rho
        self.state.U[:, :, self.state.RHOU_IDX] = self.inputs.BC_inlet_north_rho * self.inputs.BC_inlet_north_u
        self.state.U[:, :, self.state.RHOV_IDX] = self.inputs.BC_inlet_north_rho * self.inputs.BC_inlet_north_v
        self.state.U[:, :, self.state.E_IDX]    = self.inputs.BC_inlet_north_p / (self.inputs.gamma - 1) \
                                                + 0.5 * self.inputs.BC_inlet_north_rho \
                                                      * (self.inputs.BC_inlet_north_u ** 2 +
                                                         self.inputs.BC_inlet_north_v ** 2)

    def set_BC_outlet_dirichlet(self):
        self.state.U = self.refBLK.get_north_ghost()

    def set_BC_inlet_riemann(self):
        pass

    def set_BC_outlet_riemann(self):
        pass


class GhostBlockSouth(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        super().__init__(inputs, BCtype, refBLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        # Calculate location of NorthWest corner
        NWx = self.refBLK.mesh.nodes.x[0, 0]
        NWy = self.refBLK.mesh.nodes.y[0, 0]
        # Calculate location of NorthEast corner
        NEx = self.refBLK.mesh.nodes.x[0, -1]
        NEy = self.refBLK.mesh.nodes.y[0, -1]

        # Get points on the outside of the block
        SWx, SWy = utils.reflect_point(NWx, NWy, NEx, NEy, xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, 0],
                                       yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, 0])
        SEx, SEy = utils.reflect_point(NWx, NWy, NEx, NEy, xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, -1],
                                       yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, -1])
        # Construct Mesh
        self.mesh = QuadMesh(self.inputs, NE=(NEx, NEy), NW=(NWx, NWy), SE=(SEx, SEy), SW=(SWx, SWy),
                             nx=self.refBLK.ny, ny=inputs.nghost)

    def set_BC_none(self):
        self.state.U = self.refBLK.neighbors.S.get_north_ghost()

    def set_BC_reflection(self):
        """
        Set reflection boundary condition on the northern face, keeps the tangential component as is and reverses the
        sign of the normal component.
        """
        state = self.refBLK.get_south_ghost()
        self.flip_wall_v_velocity(state, self.refBLK.mesh.get_south_face_angle() - np.pi / 2)
        self.state.U = state

    def set_BC_slipwall(self):
        """
        Set slipwall boundary condition on the southern face, keeps the tangential component as is and zeros the
        normal component.
        """
        state = self.refBLK.get_south_ghost()
        self.zero_wall_v_velocity(state, self.refBLK.mesh.get_south_face_angle() - np.pi / 2)
        self.state.U = state

    def set_BC_inlet_dirichlet(self):
        """
        Set dirichlet inlet boundary condition on the southern face.
        """
        self.state.U[:, :, self.state.RHO_IDX]  = self.inputs.BC_inlet_south_rho
        self.state.U[:, :, self.state.RHOU_IDX] = self.inputs.BC_inlet_south_rho * self.inputs.BC_inlet_south_u
        self.state.U[:, :, self.state.RHOV_IDX] = self.inputs.BC_inlet_south_rho * self.inputs.BC_inlet_south_v
        self.state.U[:, :, self.state.E_IDX]    = self.inputs.BC_inlet_south_p / (self.inputs.gamma - 1) \
                                                + 0.5 * self.inputs.BC_inlet_south_rho \
                                                      * (self.inputs.BC_inlet_south_u ** 2 +
                                                         self.inputs.BC_inlet_south_v ** 2)

    def set_BC_outlet_dirichlet(self):
        self.state.U = self.refBLK.get_south_ghost()

    def set_BC_inlet_riemann(self):
        pass

    def set_BC_outlet_riemann(self):
        pass
