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
from __future__ import annotations

import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
from typing import Union
from typing import TYPE_CHECKING
from pyhype.utils.utils import SidePropertyDict

if TYPE_CHECKING:
    from pyhype.mesh.quad_mesh import QuadMesh
    from pyhype.solvers.base import SolverConfig


# Define quadrature sets
_QUAD_1 = {0: 2}
_QUAD_2 = {-1 / np.sqrt(3): 1, 1 / np.sqrt(3): 1}
_QUAD_3 = {-np.sqrt(3 / 5): 5 / 9, 0: 8 / 9, np.sqrt(3 / 5): 5 / 9}

_QUADS = {1: _QUAD_1, 2: _QUAD_2, 3: _QUAD_3}


class QuadraturePoint:
    def __init__(
        self,
        x: np.ndarray = None,
        y: np.ndarray = None,
        w: Union[np.ndarray, float, int] = None,
    ):
        self.x = x
        self.y = y
        self.w = w


class QuadraturePointData(SidePropertyDict):
    def __init__(self, config: SolverConfig, refMESH: QuadMesh):
        super().__init__(*self._make_quadrature(config, refMESH))

    @staticmethod
    def _transform(p1: np.ndarray, p2: np.ndarray, p: float):
        return 0.5 * ((p2 - p1) * p + (p2 + p1))

    def _make_quadrature(self, config: SolverConfig, refMESH: QuadMesh):

        xNE, yNE = refMESH.get_NE_vertices()
        xNW, yNW = refMESH.get_NW_vertices()
        xSE, ySE = refMESH.get_SE_vertices()
        xSW, ySW = refMESH.get_SW_vertices()

        E = tuple(
            QuadraturePoint(
                self._transform(xNE, xSE, p), self._transform(yNE, ySE, p), w
            )
            for p, w in _QUADS[config.fvm_num_quadrature_points].items()
        )

        W = tuple(
            QuadraturePoint(
                self._transform(xNW, xSW, p), self._transform(yNW, ySW, p), w
            )
            for p, w in _QUADS[config.fvm_num_quadrature_points].items()
        )

        N = tuple(
            QuadraturePoint(
                self._transform(xNE, xNW, p), self._transform(yNE, yNW, p), w
            )
            for p, w in _QUADS[config.fvm_num_quadrature_points].items()
        )

        S = tuple(
            QuadraturePoint(
                self._transform(xSE, xSW, p), self._transform(ySE, ySW, p), w
            )
            for p, w in _QUADS[config.fvm_num_quadrature_points].items()
        )

        return E, W, N, S
