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


def rotate(theta: float,
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

    for array in arrays:

        u = array[:, :, 1] * np.cos(theta) + array[:, :, 2] * np.sin(theta)
        v = array[:, :, 2] * np.cos(theta) - array[:, :, 1] * np.sin(theta)

        array[:, :, 1] = u
        array[:, :, 2] = v


def unrotate(theta: float,
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

    for array in arrays:
        u = array[:, :, 1] * np.cos(theta) - array[:, :, 2] * np.sin(theta)
        v = array[:, :, 2] * np.cos(theta) + array[:, :, 1] * np.sin(theta)

        array[:, :, 1] = u
        array[:, :, 2] = v

