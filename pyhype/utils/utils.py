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
from dataclasses import dataclass
from typing import Union, Callable, TYPE_CHECKING, Any

import numba as nb

if TYPE_CHECKING:
    from pyhype.states.base import State

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


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


@dataclass
class SidePropertyContainer:
    E: Any
    W: Any
    N: Any
    S: Any


@dataclass
class CornerPropertyContainer:
    NE: Any
    NW: Any
    SE: Any
    SW: Any


@dataclass
class FullPropertyContainer(SidePropertyContainer, CornerPropertyContainer):
    pass


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
