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

import numpy as np
from abc import abstractmethod
from pyHype.states import ConservativeState
from pyHype.utils import utils


class GhostBlock:
    def __init__(self, inputs, BCtype: str, ref_BLK):

        self.BCtype = BCtype
        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.nghost = inputs.nghost
        self.ref_BLK = ref_BLK

        self.state = None
        self.theta = None
        self.x = None
        self.y = None
        self.xc = None
        self.yc = None

        # Assign the BCset method to avoid checking type everytime
        if self.BCtype == 'None':
            self.set_BC = self.set_BC_none
        elif self.BCtype == 'Outflow':
            self.set_BC = self.set_BC_outflow
        elif self.BCtype == 'Reflection':
            self.set_BC = self.set_BC_reflection



    def __getitem__(self, index):
        return self.state.U[index]


    """
    MODIFY
    
    def get_mesh_from_east(self):
        self.x = self.ref_BLK.boundaryBLK.E.x[:, 1]

    def get_mesh_from_west(self):
        self.x = self.ref_BLK.boundaryBLK.W.x[:, -2]

    def get_mesh_from_north(self):
        self.x = self.ref_BLK.boundaryBLK.N.x[1, :]

    def get_mesh_from_south(self):
        self.x = self.ref_BLK.boundaryBLK.S.x[-2, :]
    """

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
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = ref_BLK.thetaE

        x = self.ref_BLK.mesh.x[:, -1]
        y = self.ref_BLK.mesh.y[:, -1]

        dx = x - self.ref_BLK.mesh.x[:, -2]
        dy = y - self.ref_BLK.mesh.y[:, -2]

        self.x[:, 0] = x + dx
        self.y[:, 0] = y + dy

        cont_x = np.concatenate((self.x, ref_BLK.mesh.x[:, -1:]), axis=1)
        cont_y = np.concatenate((self.y, ref_BLK.mesh.y[:, -1:]), axis=1)

        xc, yc = ref_BLK.mesh.get_centroid(cont_x, cont_y)

        self.ref_BLK.mesh.xc[1:-1, -1] = xc.reshape(-1, )
        self.ref_BLK.mesh.yc[1:-1, -1] = yc.reshape(-1, )

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.E.get_west_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_east_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.ref_BLK.get_east_ghost()
        utils.rotate(self.ref_BLK.mesh.thetax[-1], state)
        state[:, :, 1] *= -1

        self.state.update(state)


class GhostBlockWest(GhostBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = ref_BLK.thetaW

        x = self.ref_BLK.mesh.x[:, 0]
        y = self.ref_BLK.mesh.y[:, 0]

        dx = x - self.ref_BLK.mesh.x[:, 1]
        dy = y - self.ref_BLK.mesh.y[:, 1]

        self.x[:, 0] = x + dx
        self.y[:, 0] = y + dy

        cont_x = np.concatenate((self.x, ref_BLK.mesh.x[:, 0:1]), axis=1)
        cont_y = np.concatenate((self.y, ref_BLK.mesh.y[:, 0:1]), axis=1)

        xc, yc = ref_BLK.mesh.get_centroid(cont_x, cont_y)

        self.ref_BLK.mesh.xc[1:-1, 0] = xc.reshape(-1, )
        self.ref_BLK.mesh.yc[1:-1, 0] = yc.reshape(-1, )

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.W.get_east_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_west_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.ref_BLK.get_west_ghost()
        utils.rotate(self.ref_BLK.mesh.thetax[0], state)
        state[:, :, 1] *= -1
        self.state.update(state)


class GhostBlockNorth(GhostBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = ref_BLK.thetaN

        x = self.ref_BLK.mesh.x[-1, :]
        y = self.ref_BLK.mesh.y[-1, :]

        dx = x - self.ref_BLK.mesh.x[-2, :]
        dy = y - self.ref_BLK.mesh.y[-2, :]

        self.x[0, :] = x + dx
        self.y[0, :] = y + dy

        cont_x = np.concatenate((self.x, ref_BLK.mesh.x[-1:, :]), axis=0)
        cont_y = np.concatenate((self.y, ref_BLK.mesh.y[-1:, :]), axis=0)

        xc, yc = ref_BLK.mesh.get_centroid(cont_x, cont_y)

        self.ref_BLK.mesh.xc[-1, 1:-1] = xc
        self.ref_BLK.mesh.yc[-1, 1:-1] = yc

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.N.get_south_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_north_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.ref_BLK.get_north_ghost()
        utils.rotate(np.pi / 2 - self.ref_BLK.mesh.thetay[-1], state)
        state[:, :, 2] *= -1
        self.state.update(state)


class GhostBlockSouth(GhostBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = ref_BLK.thetaS

        x = self.ref_BLK.mesh.x[0, :]
        y = self.ref_BLK.mesh.y[0, :]

        dx = x - self.ref_BLK.mesh.x[1, :]
        dy = y - self.ref_BLK.mesh.y[1, :]

        self.x[0, :] = x + dx
        self.y[0, :] = y + dy

        cont_x = np.concatenate((self.x, ref_BLK.mesh.x[0:1, :]), axis=0)
        cont_y = np.concatenate((self.y, ref_BLK.mesh.y[0:1, :]), axis=0)

        xc, yc = ref_BLK.mesh.get_centroid(cont_x, cont_y)

        self.ref_BLK.mesh.xc[0, 1:-1] = xc
        self.ref_BLK.mesh.yc[0, 1:-1] = yc

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.S.get_north_ghost()
        self.state.set_vars_from_state()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_south_edge()
        self.state.set_vars_from_state()

    def set_BC_reflection(self):
        state = self.ref_BLK.get_south_ghost()
        utils.rotate(np.pi / 2 - self.ref_BLK.mesh.thetay[0], state)
        state[:, :, 2] *= -1
        self.state.update(state)
