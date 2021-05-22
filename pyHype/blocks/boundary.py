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
    def __init__(self, inputs, type_: str, ref_BLK):

        self._type = type_
        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.ref_BLK = ref_BLK

        self._state = None
        self.theta = None
        self.x = None
        self.y = None

    def __getitem__(self, index):
        return self.state.U[index]

    @property
    def state(self):
        return self._state

    def set(self) -> None:
        if self._type == 'None':
            self.set_BC_none()
        elif self._type == 'Outflow':
            self.set_BC_outflow()
        elif self._type == 'Reflection':
            self.set_BC_reflection()

    @abstractmethod
    def from_ref_U(self):
        pass

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
        self._state = ConservativeState(inputs, nx=self.nx, ny=1)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = ref_BLK.thetaN

        x = self.ref_BLK.mesh.x[-1, :]
        y = self.ref_BLK.mesh.y[-1, :]

        dx = np.absolute(x - self.ref_BLK.mesh.x[-2, :])
        dy = np.absolute(y - self.ref_BLK.mesh.y[-2, :])

        self.x[0, :] = x + dx * np.cos(self.theta)
        self.y[0, :] = y + dy * np.sin(self.theta)

    def from_ref_U(self):
        return self.ref_BLK.state.U[-1, :, :].reshape(1, self.inputs.nx, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.N.get_south_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 2] *= -1


class BoundaryBlockSouth(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=self.nx, ny=1)

        self.x = np.zeros((1, self.inputs.nx))
        self.y = np.zeros((1, self.inputs.nx))

        self.theta = ref_BLK.thetaS

        x = self.ref_BLK.mesh.x[0, :]
        y = self.ref_BLK.mesh.y[0, :]

        dx = np.absolute(x - self.ref_BLK.mesh.x[1, :])
        dy = np.absolute(y - self.ref_BLK.mesh.y[1, :])

        self.x[0, :] = x + dx * np.cos(self.theta)
        self.y[0, :] = y + dy * np.sin(self.theta)



    def from_ref_U(self):
        return self.ref_BLK.state.U[0, :, :].reshape(1, self.inputs.nx, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_north_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 2] *= -1


class BoundaryBlockEast(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=1, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = ref_BLK.thetaE

        x = self.ref_BLK.mesh.x[:, -1]
        y = self.ref_BLK.mesh.y[:, -1]

        dx = np.absolute(x - self.ref_BLK.mesh.x[:, -2])
        dy = np.absolute(y - self.ref_BLK.mesh.y[:, -2])

        self.x[:, 0] = x + dx * np.cos(self.theta)
        self.y[:, 0] = y + dy * np.sin(self.theta)

    def from_ref_U(self):
        return self.ref_BLK.state.U[:, -1, :].reshape(-1, 1, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_west_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 1] *= -1


class BoundaryBlockWest(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=1, ny=self.ny)

        self.x = np.zeros((self.inputs.ny, 1))
        self.y = np.zeros((self.inputs.ny, 1))

        self.theta = ref_BLK.thetaW

        x = self.ref_BLK.mesh.x[:, 0]
        y = self.ref_BLK.mesh.y[:, 0]

        dx = np.absolute(x - self.ref_BLK.mesh.x[:, 1])
        dy = np.absolute(y - self.ref_BLK.mesh.y[:, 1])

        self.x[:, 0] = x + dx * np.cos(self.theta)
        self.y[:, 0] = y + dy * np.sin(self.theta)

    def from_ref_U(self):
        return self.ref_BLK.state.U[:, 0, :].reshape(-1, 1, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_east_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 1] *= -1
