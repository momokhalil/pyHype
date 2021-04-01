import numba
from numba import float32
from abc import ABC, abstractmethod



class FluxFunction(ABC):
    def __init__(self, input_):
        self._input = input_
        self._L = None
        self._R = None
        self.g = input_.gamma
        self.nx = input_.nx
        self.ny = input_.ny
        self.n = input_.n

    def set_left_state(self, UL):
        self._L = UL

    def set_right_state(self, UR):
        self._R = UR

    def dULR(self):
        return self._L.U - self._R.U

    def L_plus_R(self):
        return self._L.U + self._R.U

    @abstractmethod
    def get_flux(self):
        pass



