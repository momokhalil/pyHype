from typing import Union
from abc import abstractmethod, ABC


class Vertices:
    def __init__(self, NE: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       NW: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       SE: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       SW: tuple[Union[float, int], Union[float, int]] = (0, 0)) -> None:
        self.NW = NW
        self.NE = NE
        self.SW = SW
        self.SE = SE


class Neighbors:
    def __init__(self, E=None,
                       W=None,
                       N=None,
                       S=None) -> None:
        self.E = E
        self.W = W
        self.N = N
        self.S = S


class BoundaryBlocks:
    def __init__(self, E: 'BoundaryBlock' = None,
                       W: 'BoundaryBlock' = None,
                       N: 'BoundaryBlock' = None,
                       S: 'BoundaryBlock' = None) -> None:
        self.E = E
        self.W = W
        self.N = N
        self.S = S


class BoundaryBlock(ABC):
    def __init__(self, inputs, type_: str):
        self.inputs = inputs
        self._idx_from_U = None
        self._state = None
        self._type = type_
        self.nx = inputs.nx
        self.ny = inputs.ny

    @property
    def state(self):
        return self._state

    @abstractmethod
    def set(self, ref_BLK):
        pass

    def from_ref_U(self, ref_BLK):
        return ref_BLK.state.U[self._idx_from_U]
