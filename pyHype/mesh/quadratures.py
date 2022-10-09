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
import pyHype.states as states

if TYPE_CHECKING:
    from pyHype.mesh.QuadMesh import QuadMesh
    from pyHype.solvers.base import ProblemInput
    from pyHype.states import State, PrimitiveState, ConservativeState


# Define quadrature sets
_QUAD_1 = {0: 2}
_QUAD_2 = {-1 / np.sqrt(3): 1, 1 / np.sqrt(3): 1}
_QUAD_3 = {-np.sqrt(3 / 5): 5 / 9, 0: 8 / 9, np.sqrt(3 / 5): 5 / 9}

_QUADS = {1: _QUAD_1, 2: _QUAD_2, 3: _QUAD_3}


class QuadraturePoint:
    def __init__(
        self,
        inputs: ProblemInput,
        x: np.ndarray = None,
        y: np.ndarray = None,
        w: Union[np.ndarray, float, int] = None,
    ):
        self.x = x
        self.y = y
        self.w = w
        if inputs.reconstruction_type == "primitive":
            self.state = states.PrimitiveState(inputs=inputs)
        elif inputs.reconstruction_type == "conservative":
            self.state = states.ConservativeState(inputs=inputs)


class QuadraturePointDataContainerBase:
    def __init__(
        self,
        dataE: Union[list, tuple] = None,
        dataW: Union[list, tuple] = None,
        dataN: Union[list, tuple] = None,
        dataS: Union[list, tuple] = None,
    ) -> None:
        self.E = None
        self.W = None
        self.N = None
        self.S = None

        if dataE and dataW and dataN and dataS:
            self.create_data(dataE, dataW, dataN, dataS)

    def create_data(
        self,
        dataE: Union[list, tuple],
        dataW: Union[list, tuple],
        dataN: Union[list, tuple],
        dataS: Union[list, tuple],
    ) -> None:
        self.E = tuple(dataE)
        self.W = tuple(dataW)
        self.N = tuple(dataN)
        self.S = tuple(dataS)


class QuadraturePointStateContainer(QuadraturePointDataContainerBase):
    def __init__(
        self,
        dataE: Union[list[State], tuple[State]] = None,
        dataW: Union[list[State], tuple[State]] = None,
        dataN: Union[list[State], tuple[State]] = None,
        dataS: Union[list[State], tuple[State]] = None,
    ) -> None:
        super().__init__(dataE, dataW, dataN, dataS)

    def update_data(
        self,
        dataE: [np.ndarray],
        dataW: [np.ndarray],
        dataN: [np.ndarray],
        dataS: [np.ndarray],
    ) -> None:
        for n, (stateE, stateW, stateN, stateS) in enumerate(
            zip(dataE, dataW, dataN, dataS)
        ):
            self.E[n].Q = stateE
            self.W[n].Q = stateW
            self.N[n].Q = stateN
            self.S[n].Q = stateS


class QuadraturePointData:
    def __init__(self, inputs: ProblemInput, refMESH: QuadMesh):
        self.E = None
        self.W = None
        self.N = None
        self.S = None
        self._make_quadrature(inputs, refMESH)

    @staticmethod
    def _transform(p1: np.ndarray, p2: np.ndarray, p: float):
        return 0.5 * ((p2 - p1) * p + (p2 + p1))

    def _make_quadrature(self, inputs: ProblemInput, refMESH: QuadMesh):

        xNE, yNE = refMESH.get_NE_vertices()
        xNW, yNW = refMESH.get_NW_vertices()
        xSE, ySE = refMESH.get_SE_vertices()
        xSW, ySW = refMESH.get_SW_vertices()

        self.E = tuple(
            QuadraturePoint(
                inputs, self._transform(xNE, xSE, p), self._transform(yNE, ySE, p), w
            )
            for p, w in _QUADS[inputs.fvm_num_quadrature_points].items()
        )

        self.W = tuple(
            QuadraturePoint(
                inputs, self._transform(xNW, xSW, p), self._transform(yNW, ySW, p), w
            )
            for p, w in _QUADS[inputs.fvm_num_quadrature_points].items()
        )

        self.N = tuple(
            QuadraturePoint(
                inputs, self._transform(xNE, xNW, p), self._transform(yNE, yNW, p), w
            )
            for p, w in _QUADS[inputs.fvm_num_quadrature_points].items()
        )

        self.S = tuple(
            QuadraturePoint(
                inputs, self._transform(xSE, xSW, p), self._transform(ySE, ySW, p), w
            )
            for p, w in _QUADS[inputs.fvm_num_quadrature_points].items()
        )

    def update_data(
        self,
        dataE: [np.ndarray],
        dataW: [np.ndarray],
        dataN: [np.ndarray],
        dataS: [np.ndarray],
    ) -> None:
        for n, (stateE, stateW, stateN, stateS) in enumerate(
            zip(dataE, dataW, dataN, dataS)
        ):
            self.E[n].state.Q = stateE
            self.W[n].state.Q = stateW
            self.N[n].state.Q = stateN
            self.S[n].state.Q = stateS
