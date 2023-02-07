from __future__ import annotations

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
import functools
from collections import UserDict
from dataclasses import dataclass
from typing import Union, Callable, TYPE_CHECKING, Any

import numba as nb

if TYPE_CHECKING:
    from pyhype.states.base import State

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


@dataclass
class Direction:
    east = 1
    west = 2
    north = 3
    south = 4
    north_east = 5
    north_west = 6
    south_east = 7
    south_west = 8


def rotate(
    theta: Union[float, np.ndarray],
    *arrays: np.ndarray,
) -> None:
    """
    Rotates a 1 * nx * 4 ndarray that represents a row of nodes from a State by theta degrees counterclockwise.
    The contents of the array may be Conservative/Primitive state variables, Fluxes, etc...

                            y
                            |
     y'                     |                      x'
        *                   |                   *
           *                |                *
              *             |             *
                 *          |          *
                    *       |       *  \
                       *    |    *      \
                          * | *    theta \
                            ------------------------------- x

    x' = x cos(theta) + y sin(theta)
    y' = y cos(theta) - x sin(theta)
    """

    if np.ndim(theta) == 3:
        theta = theta[:, :, 0]
    elif np.ndim(theta) > 3:
        raise RuntimeError("theta cannot have more than 3 dimensions.")

    for array in arrays:
        u, v = rotate_JIT(array, theta)
        array[:, :, 1] = u
        array[:, :, 2] = v


@nb.njit(cache=True)
def rotate_JIT(array, theta):
    u = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            _theta = theta[i, j]
            _u = array[i, j, 1]
            _v = array[i, j, 2]
            s = np.sin(_theta)
            c = np.cos(_theta)
            u[i, j] = _u * c + _v * s
            v[i, j] = _v * c - _u * s

    return u, v


def rotate90(*arrays: np.ndarray) -> None:
    """
    Rotates a 1 * nx * 4 ndarray that represents a row of nodes from a State by ninety degrees counterclockwise.
    The contents of the array may be Conservative/Primitive state variables, Fluxes, etc...

                         - = y  ,  * = x'
                                *
                                |
                                *
                                |
                                *
                                |
                                * <------
                                | theta |
                                *       |
    y' ************************* --------------------------- x

    x' = y
    y' = -x
    """

    for array in arrays:

        u = array[:, :, 2].copy()
        v = array[:, :, 1].copy()

        array[:, :, 1], array[:, :, 2] = u, -v


def unrotate(
    theta: float,
    *arrays: np.ndarray,
) -> None:
    """
    Rotates a 1 * nx * 4 ndarray that represents a row of nodes from a State by theta degrees clockwise. Basically the
    inverse of rotate_row(). The contents of the array may be Conservative/Primitive state variables, Fluxes, etc...

                            y
                            |
     y'                     |                      x'
        *                   |                   *
           *                |                *
              *             |             *
                 *          |          *
                    *       |       *  \
                       *    |    *      \
                          * | *    theta \
                            ------------------------------- x

    x = x' cos(theta) - y' sin(theta)
    y = y' cos(theta) + x' sin(theta)
    """

    if np.ndim(theta) == 3:
        theta = theta[:, :, 0]
    elif np.ndim(theta) > 3:
        raise RuntimeError("theta cannot have more than 3 dimensions.")

    for array in arrays:
        u, v = unrotate_JIT(array, theta)
        array[:, :, 1] = u
        array[:, :, 2] = v


@nb.njit(cache=True)
def unrotate_JIT(array, theta):
    u = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            _theta = theta[i, j]
            _u = array[i, j, 1]
            _v = array[i, j, 2]
            s = np.sin(_theta)
            c = np.cos(_theta)
            u[i, j] = _u * c - _v * s
            v[i, j] = _v * c + _u * s

    return u, v


def unrotate90(*arrays: np.ndarray) -> None:
    """
    Rotates a 1 * nx * 4 ndarray that represents a row of nodes from a State by ninety degrees clockwise.
    The contents of the array may be Conservative/Primitive state variables, Fluxes, etc...

                         - = y  ,  * = x'
                                *
                                |
                                *
                                |
                                *
                                |
                        ------> *
                        | theta |
                        |       *
    y' ************************* --------------------------- x

    x = -y'
    y = x'
    """

    for array in arrays:
        u = array[:, :, 2].copy()
        v = array[:, :, 1].copy()
        array[:, :, 1], array[:, :, 2] = -u, v


def reflect_point(
    x1: float, y1: float, x2: float, y2: float, xr: float, yr: float
) -> [float]:

    if y1 == y2:
        return xr, 2 * y1 - yr
    if x1 == x2:
        return 2 * x1 - xr, yr
    m = (y1 - y2) / (x1 - x2)
    b = 0.5 * (y1 + y2 - m * (x1 + x2))

    xp = ((1 - m**2) * xr + 2 * m * yr - 2 * m * b) / (1 + m**2)
    yp = (2 * m * xr - (1 - m**2) * yr + 2 * b) / (1 + m**2)

    return xp, yp


def cache(func: Callable):
    @functools.wraps(func)
    def _wrapper(self, *args, **kwargs):
        if func.__name__ not in self.cache.keys():
            self.cache[func.__name__] = func(self, *args, **kwargs)
        return self.cache[func.__name__]

    return _wrapper


class Cache:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return functools.partial(self, instance)

    def __call__(self, *args, **kwargs):
        instance = args[0]
        if self.func not in instance.cache.keys():
            instance.cache[self.func] = self.func(*args, **kwargs)
        return instance.cache[self.func]


class SidePropertyDict(UserDict):
    def __init__(self, E: Any, W: Any, N: Any, S: Any):
        _items = {
            Direction.east: E,
            Direction.west: W,
            Direction.north: N,
            Direction.south: S,
        }
        super().__init__(_items)

        self._E = E
        self._W = W
        self._N = N
        self._S = S

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E: Any):
        self._E = self.data[Direction.east] = E

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, W: Any):
        self._W = self.data[Direction.west] = W

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N: Any):
        self._N = self.data[Direction.north] = N

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, S: Any):
        self._S = self.data[Direction.south] = S


class CornerPropertyDict(UserDict):
    def __init__(self, NE: Any, NW: Any, SE: Any, SW: Any):
        _items = {
            Direction.north_east: NE,
            Direction.north_west: NW,
            Direction.south_east: SE,
            Direction.south_west: SW,
        }
        super().__init__(_items)

        self._NE = NE
        self._NW = NW
        self._SE = SE
        self._SW = SW

    @property
    def NE(self):
        return self._NE

    @NE.setter
    def NE(self, NE: Any):
        self._NE = self.data[Direction.north_east] = NE

    @property
    def NW(self):
        return self._NW

    @NW.setter
    def NW(self, NW: Any):
        self._NW = self.data[Direction.north_west] = NW

    @property
    def SE(self):
        return self._SE

    @SE.setter
    def SE(self, SE: Any):
        self._SE = self.data[Direction.south_east] = SE

    @property
    def SW(self):
        return self._SW

    @SW.setter
    def SW(self, SW: Any):
        self._SW = self.data[Direction.south_west] = SW


class FullPropertyDict(UserDict):
    def __init__(
        self, E: Any, W: Any, N: Any, S: Any, NE: Any, NW: Any, SE: Any, SW: Any
    ):
        _items = {
            Direction.east: E,
            Direction.west: W,
            Direction.north: N,
            Direction.south: S,
            Direction.north_east: NE,
            Direction.north_west: NW,
            Direction.south_east: SE,
            Direction.south_west: SW,
        }
        super().__init__(_items)

        self._E = E
        self._W = W
        self._N = N
        self._S = S
        self._NE = NE
        self._NW = NW
        self._SE = SE
        self._SW = SW

    @property
    def E(self):
        return self._E

    @E.setter
    def E(self, E: Any):
        self._E = self.data[Direction.east] = E

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, W: Any):
        self._W = self.data[Direction.west] = W

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N: Any):
        self._N = self.data[Direction.north] = N

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, S: Any):
        self._S = self.data[Direction.south] = S

    @property
    def NE(self):
        return self._NE

    @NE.setter
    def NE(self, NE: Any):
        self._NE = self.data[Direction.north_east] = NE

    @property
    def NW(self):
        return self._NW

    @NW.setter
    def NW(self, NW: Any):
        self._NW = self.data[Direction.north_west] = NW

    @property
    def SE(self):
        return self._SE

    @SE.setter
    def SE(self, SE: Any):
        self._SE = self.data[Direction.south_east] = SE

    @property
    def SW(self):
        return self._SW

    @SW.setter
    def SW(self, SW: Any):
        self._SW = self.data[Direction.south_west] = SW


class NumpySlice:
    def __init__(self):
        pass

    @staticmethod
    def all():
        return np.s_[...]

    @staticmethod
    def col(index: int):
        return np.s_[:, index, None, :]

    @staticmethod
    def row(index: int):
        return np.s_[index, None, :, :]

    @staticmethod
    def cols(start: int = None, end: int = None):
        if start and end:
            return np.s_[:, start:end, :]
        if start:
            return np.s_[:, start:, :]
        if end:
            return np.s_[:, :end, :]

    @staticmethod
    def rows(start: int = None, end: int = None):
        if start and end:
            return np.s_[start:end, :, :]
        if start:
            return np.s_[start:, :, :]
        if end:
            return np.s_[:end, :, :]

    @classmethod
    def east_boundary(cls):
        return cls.col(-1)

    @classmethod
    def west_boundary(cls):
        return cls.col(0)

    @classmethod
    def north_boundary(cls):
        return cls.row(-1)

    @classmethod
    def south_boundary(cls):
        return cls.row(0)

    @classmethod
    def east_face(cls):
        return cls.cols(start=1)

    @classmethod
    def west_face(cls):
        return cls.cols(end=-1)

    @classmethod
    def north_face(cls):
        return cls.rows(start=1)

    @classmethod
    def south_face(cls):
        return cls.rows(end=-1)
