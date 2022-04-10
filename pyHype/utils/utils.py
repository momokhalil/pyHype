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

import functools
import numpy as np
from typing import Union, Callable


def rotate(theta: Union[float, np.ndarray],
           *arrays: Union[np.ndarray, list[np.ndarray]],
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
        raise RuntimeError('theta cannot have more than 3 dimensions.')

    for array in arrays:
        u = array[:, :, 1] * np.cos(theta) + array[:, :, 2] * np.sin(theta)
        v = array[:, :, 2] * np.cos(theta) - array[:, :, 1] * np.sin(theta)

        array[:, :, 1] = u
        array[:, :, 2] = v


def rotate90(*arrays: Union[np.ndarray]) -> None:
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
        v = -array[:, :, 1].copy()

        array[:, :, 1], array[:, :, 2] = u, v


def unrotate(theta: float,
             *arrays: Union[np.ndarray, list[np.ndarray]],
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
        raise RuntimeError('theta cannot have more than 3 dimensions.')

    for array in arrays:
        u = array[:, :, 1] * np.cos(theta) - array[:, :, 2] * np.sin(theta)
        v = array[:, :, 2] * np.cos(theta) + array[:, :, 1] * np.sin(theta)

        array[:, :, 1] = u
        array[:, :, 2] = v

def unrotate90(*arrays: Union[np.ndarray]) -> None:
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
        u = -array[:, :, 2].copy()
        v = array[:, :, 1].copy()
        array[:, :, 1], array[:, :, 2] = u, v


def reflect_point(x1: float,
                  y1: float,
                  x2: float,
                  y2: float,
                  xr: float,
                  yr: float
                  ) -> [float]:

    if y1 == y2:
        return xr, 2 * y1 - yr
    elif x1 == x2:
        return 2 * x1 - xr, yr
    else:
        m = (y1 - y2) / (x1 - x2)
        b = 0.5 * (y1 + y2 - m * (x1 + x2))

        xp = ((1 - m ** 2) * xr + 2 * m * yr - 2 * m * b) / (1 + m ** 2)
        yp = (2 * m * xr - (1 - m ** 2) * yr + 2 * b) / (1 + m ** 2)

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


class DirectionalContainerBase:
    def __init__(self,
                 east_obj: object = None,
                 west_obj: object = None,
                 north_obj: object = None,
                 south_obj: object = None):
        self.E = east_obj
        self.W = west_obj
        self.N = north_obj
        self.S = south_obj
