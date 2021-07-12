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
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from abc import abstractmethod
from pyHype.states import ConservativeState
from pyHype.utils import utils
from pyHype.mesh.base import Mesh


class GhostBlock:
    def __init__(self, inputs, BCtype: str, refBLK):

        self.BCtype = BCtype
        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.nghost = inputs.nghost
        self.refBLK = refBLK

        self.state = None
        self.theta = None
        self.x = None
        self.y = None
        self.xc = None
        self.yc = None

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
    def __init__(self, inputs, BCtype, refBLK):

        super().__init__(inputs, BCtype, refBLK)

        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny + 1, 1))
        self.y = np.zeros((self.inputs.ny + 1, 1))

        self.theta = refBLK.thetaE

        NWx = self.refBLK.mesh.nodes.x[-1, -1]
        NWy = self.refBLK.mesh.nodes.y[-1, -1]

        SWx = self.refBLK.mesh.nodes.x[0, -1]
        SWy = self.refBLK.mesh.nodes.y[0, -1]

        dx_NE = NWx - self.refBLK.mesh.nodes.x[-1, -1 - self.inputs.nghost]
        dy_NE = NWy - self.refBLK.mesh.nodes.y[-1, -1 - self.inputs.nghost]

        dx_SE = SWx - self.refBLK.mesh.nodes.x[0, -1 - self.inputs.nghost]
        dy_SE = SWy - self.refBLK.mesh.nodes.y[0, -1 - self.inputs.nghost]

        self.mesh = Mesh(self.inputs,
                         NE=(NWx + dx_NE, NWy - dy_NE),
                         NW=(NWx, NWy),
                         SE=(SWx + dx_SE, SWy - dy_SE),
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
        state = self.refBLK.get_east_ghost()
        utils.rotate(self.refBLK.mesh.thetax[-1], state)
        state[:, :, 1] *= -1

        self.state.update(state)


class GhostBlockWest(GhostBlock):
    def __init__(self, inputs, BCtype, refBLK):

        super().__init__(inputs, BCtype, refBLK)

        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny + 1, 1))
        self.y = np.zeros((self.inputs.ny + 1, 1))

        self.theta = refBLK.thetaW

        NEx = self.refBLK.mesh.nodes.x[-1, 0]
        NEy = self.refBLK.mesh.nodes.y[-1, 0]

        SEx = self.refBLK.mesh.nodes.x[0, 0]
        SEy = self.refBLK.mesh.nodes.y[0, 0]

        dx_NW = NEx - self.refBLK.mesh.nodes.x[-1, self.inputs.nghost]
        dy_NW = NEy - self.refBLK.mesh.nodes.y[-1, self.inputs.nghost]

        dx_SW = SEx - self.refBLK.mesh.nodes.x[0, self.inputs.nghost]
        dy_SW = SEy - self.refBLK.mesh.nodes.y[0, self.inputs.nghost]

        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NEx + dx_NW, NEy - dy_NW),
                         SE=(SEx, SEy),
                         SW=(SEx + dx_SW, SEy - dy_SW),
                         nx=inputs.nghost,
                         ny=self.refBLK.ny)



    def set_BC_none(self):
        self.state.U = self.refBLK.neighbors.W.get_east_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.refBLK.get_west_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.refBLK.get_west_ghost()
        utils.rotate(self.refBLK.mesh.thetax[0], state)
        state[:, :, 1] *= -1
        self.state.update(state)


class GhostBlockNorth(GhostBlock):
    def __init__(self, inputs, BCtype, refBLK):

        super().__init__(inputs, BCtype, refBLK)

        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx + 1))
        self.y = np.zeros((1, self.inputs.nx + 1))

        self.theta = refBLK.thetaN

        SWx = self.refBLK.mesh.nodes.x[-1, 0]
        SWy = self.refBLK.mesh.nodes.y[-1, 0]

        SEx = self.refBLK.mesh.nodes.x[-1, -1]
        SEy = self.refBLK.mesh.nodes.y[-1, -1]

        dx_NW = SWx - self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, 0]
        dy_NW = SWy - self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, 0]

        dx_NE = SEx - self.refBLK.mesh.nodes.x[-1 - self.inputs.nghost, -1]
        dy_NE = SEy - self.refBLK.mesh.nodes.y[-1 - self.inputs.nghost, -1]

        self.mesh = Mesh(self.inputs,
                         NE=(SEx - dx_NE, SEy + dy_NE),
                         NW=(SWx - dx_NW, SWy + dy_NW),
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
        state = self.refBLK.get_north_ghost()
        utils.rotate(np.pi / 2 - self.refBLK.mesh.thetay[-1], state)
        state[:, :, 2] *= -1
        self.state.update(state)


class GhostBlockSouth(GhostBlock):
    def __init__(self, inputs, BCtype, refBLK):

        super().__init__(inputs, BCtype, refBLK)
        
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx + 1))
        self.y = np.zeros((1, self.inputs.nx + 1))

        self.theta = refBLK.thetaS

        NWx = self.refBLK.mesh.nodes.x[0, 0]
        NWy = self.refBLK.mesh.nodes.y[0, 0]

        NEx = self.refBLK.mesh.nodes.x[0, -1]
        NEy = self.refBLK.mesh.nodes.y[0, -1]

        dx_SW = NWx - self.refBLK.mesh.nodes.x[self.inputs.nghost, 0]
        dy_SW = NWy - self.refBLK.mesh.nodes.y[self.inputs.nghost, 0]

        dx_SE = NEx - self.refBLK.mesh.nodes.x[self.inputs.nghost, -1]
        dy_SE = NEy - self.refBLK.mesh.nodes.y[self.inputs.nghost, -1]

        self.mesh = Mesh(self.inputs,
                         NE=(NEx, NEy),
                         NW=(NWx, NWy),
                         SE=(NEx - dx_SE, NEy + dy_SE),
                         SW=(NWx - dx_SW, NWy + dy_SW),
                         nx=self.refBLK.ny,
                         ny=inputs.nghost)


    def set_BC_none(self):
        self.state.U = self.refBLK.neighbors.S.get_north_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.refBLK.get_south_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.refBLK.get_south_ghost()
        utils.rotate(np.pi / 2 - self.refBLK.mesh.thetay[0], state)
        state[:, :, 2] *= -1
        self.state.update(state)
