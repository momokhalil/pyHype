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
        self.x = self.refBLK.boundaryBLK.E.x[:, 1]

    def get_mesh_from_west(self):
        self.x = self.refBLK.boundaryBLK.W.x[:, -2]

    def get_mesh_from_north(self):
        self.x = self.refBLK.boundaryBLK.N.x[1, :]

    def get_mesh_from_south(self):
        self.x = self.refBLK.boundaryBLK.S.x[-2, :]
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
    def __init__(self, inputs, type_, refBLK):
        super().__init__(inputs, type_, refBLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = refBLK.thetaE

        self.x[:, 0] = 2 * self.refBLK.mesh.x[:, -1] - self.refBLK.mesh.x[:, -2]
        self.y[:, 0] = 2 * self.refBLK.mesh.y[:, -1] - self.refBLK.mesh.y[:, -2]

        cont_x = np.concatenate((self.x, refBLK.mesh.x[:, -1:]), axis=1)
        cont_y = np.concatenate((self.y, refBLK.mesh.y[:, -1:]), axis=1)

        xc, yc = refBLK.mesh.get_centroid(cont_x, cont_y)

        self.refBLK.mesh.xc[1:-1, -1] = xc.reshape(-1, )
        self.refBLK.mesh.yc[1:-1, -1] = yc.reshape(-1, )

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
    def __init__(self, inputs, type_, refBLK):
        super().__init__(inputs, type_, refBLK)
        self.state = ConservativeState(inputs, nx=self.nghost, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = refBLK.thetaW

        self.x[:, 0] = 2 * self.refBLK.mesh.x[:, 0] - self.refBLK.mesh.x[:, 1]
        self.y[:, 0] = 2 * self.refBLK.mesh.y[:, 0] - self.refBLK.mesh.y[:, 1]

        cont_x = np.concatenate((self.x, refBLK.mesh.x[:, 0:1]), axis=1)
        cont_y = np.concatenate((self.y, refBLK.mesh.y[:, 0:1]), axis=1)

        xc, yc = refBLK.mesh.get_centroid(cont_x, cont_y)

        self.refBLK.mesh.xc[1:-1, 0] = xc.reshape(-1, )
        self.refBLK.mesh.yc[1:-1, 0] = yc.reshape(-1, )

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
    def __init__(self, inputs, type_, refBLK):
        super().__init__(inputs, type_, refBLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = refBLK.thetaN

        self.x[0, :] = 2 * self.refBLK.mesh.x[-1, :] - self.refBLK.mesh.x[-2, :]
        self.y[0, :] = 2 * self.refBLK.mesh.y[-1, :] - self.refBLK.mesh.y[-2, :]

        cont_x = np.concatenate((self.x, refBLK.mesh.x[-1:, :]), axis=0)
        cont_y = np.concatenate((self.y, refBLK.mesh.y[-1:, :]), axis=0)

        xc, yc = refBLK.mesh.get_centroid(cont_x, cont_y)

        self.refBLK.mesh.xc[-1, 1:-1] = xc
        self.refBLK.mesh.yc[-1, 1:-1] = yc

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
    def __init__(self, inputs, type_, refBLK):
        super().__init__(inputs, type_, refBLK)
        self.state = ConservativeState(inputs, nx=self.nx, ny=self.nghost)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = refBLK.thetaS

        self.x[0, :] = 2 * self.refBLK.mesh.x[0, :] - self.refBLK.mesh.x[1, :]
        self.y[0, :] = 2 * self.refBLK.mesh.y[0, :] - self.refBLK.mesh.y[1, :]

        cont_x = np.concatenate((self.x, refBLK.mesh.x[0:1, :]), axis=0)
        cont_y = np.concatenate((self.y, refBLK.mesh.y[0:1, :]), axis=0)

        xc, yc = refBLK.mesh.get_centroid(cont_x, cont_y)

        self.refBLK.mesh.xc[0, 1:-1] = xc
        self.refBLK.mesh.yc[0, 1:-1] = yc

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
