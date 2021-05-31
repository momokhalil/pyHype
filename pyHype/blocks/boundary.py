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


class BoundaryBlock:
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


class BoundaryBlockNorth(BoundaryBlock):
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

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.N.get_south_ghost()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_north_edge()

    def set_BC_reflection(self):
        self.state.U = self.ref_BLK.get_north_ghost()
        self.state.U[:, :, 2] *= -1


class BoundaryBlockSouth(BoundaryBlock):
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

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.S.get_north_ghost()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_south_edge()

    def set_BC_reflection(self):
        self.state.U = self.ref_BLK.get_south_ghost()
        self.state.U[:, :, 2] *= -1


class BoundaryBlockEast(BoundaryBlock):
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

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.E.get_west_ghost()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_east_edge()

    def set_BC_reflection(self):
        self.state.U = self.ref_BLK.get_east_ghost()
        self.state.U[:, :, 1] *= -1


class BoundaryBlockWest(BoundaryBlock):
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

    def set_BC_none(self):
        self.state.U = self.ref_BLK.neighbors.W.get_east_ghost()

    def set_BC_outflow(self):
        self.state.U[:, :, :] = self.ref_BLK.get_west_edge()

    def set_BC_reflection(self):
        self.state.U = self.ref_BLK.get_west_ghost()
        self.state.U[:, :, 1] *= -1
