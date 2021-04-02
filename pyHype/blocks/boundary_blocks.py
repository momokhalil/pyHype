import numpy as np
from pyHype.states import ConservativeState
from pyHype.blocks.base import BoundaryBlock


class BoundaryBlockNorth(BoundaryBlock):
    def __init__(self, inputs, type_):
        super().__init__(inputs, type_)
        self._idx_from_U = slice(4 * self.nx * (self.ny - 1), 4 * self.nx * self.ny)
        self._state = ConservativeState(inputs, self.nx)

    def set(self, ref_BLK) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.N.get_south_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockSouth(BoundaryBlock):
    def __init__(self, inputs, type_):
        super().__init__(inputs, type_)
        self._idx_from_U = slice(0, 4 * self.nx)
        self._state = ConservativeState(inputs, self.nx)

    def set(self, ref_BLK) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.S.get_north_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockEast(BoundaryBlock):
    def __init__(self, inputs, type_):
        super().__init__(inputs, type_)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(inputs, self.ny)

        for j in range(1, self.ny + 1):
            iF = 4 * self.nx * j - 4
            iE = 4 * self.nx * j
            self._idx_from_U[4 * j - 4:4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self, ref_BLK) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.E.get_west_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1

class BoundaryBlockWest(BoundaryBlock):
    def __init__(self, inputs, type_):
        super().__init__(inputs, type_)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(inputs, self.ny)

        for j in range(1, self.ny + 1):
            iF = (4 * j - 4) + 4 * (j - 1) * (self.nx - 1)
            iE = (4 * j - 0) + 4 * (j - 1) * (self.ny - 1)
            self._idx_from_U[4 * j - 4: 4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self, ref_BLK) -> None:
        if self._type == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.W.get_east_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1
