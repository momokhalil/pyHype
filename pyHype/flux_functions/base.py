import numba
from numba import float32
from abc import ABC, abstractmethod



class FluxFunction(ABC):
    def __init__(self, inputs):
        self.inputs = inputs
        self._L = None
        self._R = None
        self.g = inputs.gamma
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.n = inputs.n

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



