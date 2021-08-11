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
from pyHype.mesh.base import Mesh
from typing import TYPE_CHECKING
from pyHype.states import ConservativeState

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput
    from pyHype.blocks.base import QuadBlock

_DEFINED_BC_ = ['explosion',
                'implosion',
                'shockbox',
                'supersonic_flood',
                'supersonic_rest',
                'subsonic_flood',
                'subsonic_rest',
                'explosion_trapezoid'
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
        self.theta = None
        self.mesh = None

        # Assign the BCset method to avoid checking type everytime
        if self.BCtype == 'None':
            self.set_BC = self.set_BC_none
        elif self.BCtype == 'Outflow':
            self.set_BC = self.set_BC_outflow
        elif self.BCtype == 'Reflection':
            self.set_BC = self.set_BC_reflection
        elif self.BCtype == 'Slipwall':
            self.set_BC = self.set_BC_slipwall
        elif self.BCtype == 'InletSupersonic':
            self.set_BC = self.set_BC_inlet_supersonic
        elif self.BCtype == 'InletSubsonic':
            self.set_BC = self.set_BC_inlet_subsonic
        else:
            raise ValueError('Boundary Condition type ' + str(self.BCtype) + ' has not been specialized.')

    def __getitem__(self, index):
        return self.state.U[index]

    def row(self, index: int) -> np.ndarray:
        """
        Return the solution stored in the index-th row of the mesh. For example, if index is 0, then the state at the
        most-bottom row of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which row needs to be returned.

        Return:
            - (np.ndarray): The numpy array containing the solution at the index-th row being returned.
        """

        return self.state.U[None, index, :, :]

    def col(self, index: int) -> np.ndarray:
        """
        Return the solution stored in the index-th column of the mesh. For example, if index is 0, then the state at the
        left-most column of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which column needs to be returned.

        Return:
            - (np.ndarray): The numpy array containing the soution at the index-th column being returned.
        """

        return self.state.U[:, None, index, :]

    def row_copy(self, index: int) -> np.ndarray:
        """
        Return the a copy of the solution stored in the index-th row of the mesh. For example, if index is 0, then the
        state at the most-bottom row of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which row needs to be returned.

        Return:
            - (np.ndarray): The numpy array containing the copy of the solution at the index-th row being returned.
        """

        return self.row(index).copy()

    def col_copy(self, index: int) -> np.ndarray:
        """
        Return the a copy of the solution stored in the index-th column of the mesh. For example, if index is 0, then
        the state at the most-bottom column of the mesh will be returned.

        Parameters:
            - index (int): The index that reperesents which column needs to be returned.

        Return:
            - (np.ndarray): The numpy array containing the copy of the solution at the index-th column being returned.
        """

        return self.col(index).copy()

    @abstractmethod
    def set_BC_none(self):
        pass

    @abstractmethod
    def set_BC_outflow(self):
        pass

    @abstractmethod
    def set_BC_reflection(self):
        pass

    @abstractmethod
    def set_BC_slipwall(self):
        pass

    @abstractmethod
    def set_BC_inlet_supersonic(self):
        pass

    @abstractmethod
    def set_BC_inlet_subsonic(self):
        pass


class GhostBlockEast(GhostBlock):
    """

    """

    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        # Call superclass contructor
        super().__init__(inputs, BCtype, refBLK)

        # Construct ConservativeState class to represent the solution on the mesh
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        # Set geometric angle
        self.theta = refBLK.thetaE

        # Calculate location of NorthWest corner
        NWx = self.refBLK.mesh.nodes.x[-1, -1]
        NWy = self.refBLK.mesh.nodes.y[-1, -1]
        # Calculate location of SouthWest corner
        SWx = self.refBLK.mesh.nodes.x[0, -1]
        SWy = self.refBLK.mesh.nodes.y[0, -1]

        # Get points on the outside of the block
        NEx, NEy = utils.reflect_point(NWx, NWy, SWx, SWy,
                                       xr=self.refBLK.mesh.nodes.x[-1, -1 - self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[-1, -1 - self.inputs.nghost])

        SEx, SEy = utils.reflect_point(NWx, NWy, SWx, SWy,
                                       xr=self.refBLK.mesh.nodes.x[0, -1 - self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[0, -1 - self.inputs.nghost])

        # Construct Mesh
        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NWx, NWy),
                         SE=(SEx, SEy),
                         SW=(SWx, SWy),
                         nx=inputs.nghost,
                         ny=self.refBLK.ny)

    def set_BC_none(self):
        self.state.update(self.refBLK.neighbors.E.get_west_ghost())

    def set_BC_outflow(self):
        self.state.update(self.refBLK.get_east_ghost())

    def set_BC_reflection(self):
        # Get state from interior domain
        state = self.refBLK.get_east_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_east_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle, state)
        # Reflect normal velocity
        state[:, :, 1] *= -1
        # Update state
        self.state.update(state)

    def set_BC_slipwall(self):
        # Get state from interior domain
        state = self.refBLK.get_east_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_east_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle, state)
        # Reflect normal velocity
        state[:, :, 1] = 0
        # Update state
        self.state.update(state)

    def set_BC_inlet_supersonic(self):
        pass

    def set_BC_inlet_subsonic(self):
        pass


class GhostBlockWest(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        # Call superclass contructor
        super().__init__(inputs, BCtype, refBLK)

        # Construct ConservativeState class to represent the solution on the mesh
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        # Set geometric angle
        self.theta = refBLK.thetaW

        # Calculate location of NorthEast corner
        NEx = self.refBLK.mesh.nodes.x[-1, 0]
        NEy = self.refBLK.mesh.nodes.y[-1, 0]
        # Calculate location of SouthEast corner
        SEx = self.refBLK.mesh.nodes.x[0, 0]
        SEy = self.refBLK.mesh.nodes.y[0, 0]

        # Get points on the outside of the block
        NWx, NWy = utils.reflect_point(NEx, NEy, SEx, SEy,
                                       xr=self.refBLK.mesh.nodes.x[-1, self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[-1, self.inputs.nghost])

        SWx, SWy = utils.reflect_point(NEx, NEy, SEx, SEy,
                                       xr=self.refBLK.mesh.nodes.x[0, self.inputs.nghost],
                                       yr=self.refBLK.mesh.nodes.y[0, self.inputs.nghost])

        # Construct Mesh
        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NWx, NWy),
                         SE=(SEx, SEy),
                         SW=(SWx, SWy),
                         nx=inputs.nghost,
                         ny=self.refBLK.ny)

    def set_BC_none(self):
        self.state.update(self.refBLK.neighbors.W.get_east_ghost())

    def set_BC_outflow(self):
        self.state.update(self.refBLK.get_west_ghost())

    def set_BC_reflection(self):
        # Get state from interior domain
        state = self.refBLK.get_west_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_west_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle, state)
        # Reflect normal velocity
        state[:, :, 1] *= -1
        # Update state
        self.state.update(state)

    def set_BC_slipwall(self):
        # Get state from interior domain
        state = self.refBLK.get_west_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_west_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle, state)
        # Reflect normal velocity
        state[:, :, 1] = 0
        # Update state
        self.state.update(state)

    def set_BC_inlet_supersonic(self):
        state = np.zeros_like(self.refBLK.get_west_ghost())
        rho = 1
        u = 2.0
        v = 0
        p = 1 / self.inputs.gamma
        state[:, :, 0] = rho
        state[:, :, 1] = rho * u
        state[:, :, 2] = rho * v
        state[:, :, 3] = p / (self.inputs.gamma - 1) + 0.5 * (u ** 2 + v ** 2) * rho

        self.state.update(state)

    def set_BC_inlet_subsonic(self):
        state = np.zeros_like(self.refBLK.get_west_ghost())
        rho = 1
        u = 0.5
        v = 0
        p = 1 / self.inputs.gamma
        state[:, :, 0] = rho
        state[:, :, 1] = rho * u
        state[:, :, 2] = rho * v
        state[:, :, 3] = p / (self.inputs.gamma - 1) + 0.5 * (u ** 2 + v ** 2) * rho

        self.state.update(state)


class GhostBlockNorth(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        # Call superclass contructor
        super().__init__(inputs, BCtype, refBLK)

        # Construct ConservativeState class to represent the solution on the mesh
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        # Set geometric angle
        self.theta = refBLK.thetaN

        # Calculate location of SouthWest corner
        SWx = self.refBLK.mesh.nodes.x[-1, 0]
        SWy = self.refBLK.mesh.nodes.y[-1, 0]
        # Calculate location of SouthEast corner
        SEx = self.refBLK.mesh.nodes.x[-1, -1]
        SEy = self.refBLK.mesh.nodes.y[-1, -1]

        # Get points on the outside of the block
        NWx, NWy = utils.reflect_point(SWx, SWy, SEx, SEy,
                                       xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, 0],
                                       yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, 0])

        NEx, NEy = utils.reflect_point(SWx, SWy, SEx, SEy,
                                       xr=self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, -1],
                                       yr=self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, -1])

        # Construct Mesh
        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NWx, NWy),
                         SE=(SEx, SEy),
                         SW=(SWx, SWy),
                         nx=self.refBLK.ny,
                         ny=inputs.nghost)

    def set_BC_none(self):
        self.state.update(self.refBLK.neighbors.N.get_south_ghost())

    def set_BC_outflow(self):
        self.state.update(self.refBLK.get_north_ghost())

    def set_BC_reflection(self):
        # Get state from interior domain
        state = self.refBLK.get_north_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_north_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle - np.pi/2, state)
        # Reflect normal velocity
        state[:, :, 2] *= -1
        # Update state
        self.state.update(state)

    def set_BC_slipwall(self):
        # Get state from interior domain
        state = self.refBLK.get_north_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_north_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle - np.pi/2, state)
        # Reflect normal velocity
        state[:, :, 2] = 0
        # Update state
        self.state.update(state)

    def set_BC_inlet_supersonic(self):
        pass

    def set_BC_inlet_subsonic(self):
        pass


class GhostBlockSouth(GhostBlock):
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        # Call superclass contructor
        super().__init__(inputs, BCtype, refBLK)

        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.theta = refBLK.thetaS

        # Calculate location of NorthWest corner
        NWx = self.refBLK.mesh.nodes.x[0, 0]
        NWy = self.refBLK.mesh.nodes.y[0, 0]
        # Calculate location of NorthEast corner
        NEx = self.refBLK.mesh.nodes.x[0, -1]
        NEy = self.refBLK.mesh.nodes.y[0, -1]

        # Get points on the outside of the block
        SWx, SWy = utils.reflect_point(NWx, NWy, NEx, NEy,
                                       xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, 0],
                                       yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, 0])

        SEx, SEy = utils.reflect_point(NWx, NWy, NEx, NEy,
                                       xr=self.refBLK.mesh.nodes.x[self.inputs.nghost, -1],
                                       yr=self.refBLK.mesh.nodes.y[self.inputs.nghost, -1])

        # Construct Mesh
        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NWx, NWy),
                         SE=(SEx, SEy),
                         SW=(SWx, SWy),
                         nx=self.refBLK.ny,
                         ny=inputs.nghost)

    def set_BC_none(self):
        self.state.update(self.refBLK.neighbors.S.get_north_ghost())

    def set_BC_outflow(self):
        self.state.update(self.refBLK.get_south_ghost())

    def set_BC_reflection(self):
        # Get state from interior domain
        state = self.refBLK.get_south_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_south_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle - np.pi/2, state)
        # Reflect normal velocity
        state[:, :, 2] *= -1
        # Update state
        self.state.update(state)

    def set_BC_slipwall(self):
        # Get state from interior domain
        state = self.refBLK.get_south_ghost()
        # Get wall angle
        wall_angle = self.refBLK.mesh.get_south_face_angle()
        # Rotate state to allign with wall
        utils.rotate(wall_angle - np.pi/2, state)
        # Reflect normal velocity
        state[:, :, 2] = 0
        # Update state
        self.state.update(state)

    def set_BC_inlet_supersonic(self):
        pass

    def set_BC_inlet_subsonic(self):
        pass
