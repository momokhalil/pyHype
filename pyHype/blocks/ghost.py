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
from pyHype.states import ConservativeState
from pyHype.utils import utils
from pyHype.mesh.base import Mesh
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pyHype.solvers.solver import ProblemInput
    from pyHype.blocks.base import QuadBlock


class GhostBlock:
    def __init__(self,
                 inputs: ProblemInput,
                 BCtype: str,
                 refBLK: QuadBlock):

        self.BCtype = BCtype
        self.inputs = inputs
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


    def __getitem__(self, index):
        return self.state.U[index]

    def row(self, index: int) -> np.ndarray:
        return self.state.U[None, index, :, :]

    def col(self, index: int) -> np.ndarray:
        return self.state.U[:, None, index, :]

    def row_copy(self, index: int) -> np.ndarray:
        return self.row(index).copy()

    def col_copy(self, index: int) -> np.ndarray:
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


class GhostBlockEast(GhostBlock):
    """

    """

    def __init__(self,
                 inputs,
                 BCtype,
                 refBLK):

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
        self.state.U = self.refBLK.neighbors.E.get_west_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.refBLK.get_east_edge()
        self.state.set_vars_from_state()

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


class GhostBlockWest(GhostBlock):
    def __init__(self,
                 inputs,
                 BCtype,
                 refBLK):

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
        self.state.update(self.refBLK.get_west_edge())

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


class GhostBlockNorth(GhostBlock):
    def __init__(self,
                 inputs,
                 BCtype,
                 refBLK):

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
        self.state.U = self.refBLK.neighbors.N.get_south_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.refBLK.get_north_edge()
        self.state.set_vars_from_state()

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


class GhostBlockSouth(GhostBlock):
    def __init__(self,
                 inputs,
                 BCtype,
                 refBLK):

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
        self.state.U = self.refBLK.neighbors.S.get_north_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.refBLK.get_south_edge()
        self.state.set_vars_from_state()

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
