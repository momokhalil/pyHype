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
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm.Gradients.least_squares import least_squares_9_point

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock


class Gradient:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs


class LeastSquares9Point:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

        self.stencilSW = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]]
        self.stencilNW = [[-2, 0], [-1, 0], [0, 0], [0, 1], [-2, 0], [-2, 1], [-1, 1]]
        self.stencilSE = [[0, 0], [1, 0], [0, -1], [0, -2], [0, -1], [1, -1], [1, -2]]
        self.stencilNE = [[0, -1], [0, -2], [-1, 0], [-2, 0], [-1, -2], [1, -1], [1, -2]]

    def __call__(self, refBLK):
        return self.least_squares_nearest_neighbor(refBLK)

    def least_squares_nearest_neighbor(self, refBLK):
        bdr = refBLK.boundary_blocks

        dQdx, dQdy = least_squares_9_point(refBLK.state.Q,
                                           bdr.E.state.Q, bdr.W.state.Q, bdr.N.state.Q, bdr.S.state.Q,
                                           refBLK.mesh.x, refBLK.mesh.y,
                                           bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                           bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                           self.inputs.nx, self.inputs.ny,
                                           self.stencilSW, self.stencilNW, self.stencilSE, self.stencilNE)
        return dQdx, dQdy


class GreenGauss:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    def __call__(self, refBLK: QuadBlock) -> None:
        return self.green_gauss(refBLK)

    @staticmethod
    def green_gauss(refBLK: QuadBlock) -> None:

        # Concatenate mesh state and ghost block states
        interfaceE, interfaceW, interfaceN, interfaceS = refBLK.get_interface_values()

        # Get each face's contribution to dUdx
        E = interfaceE * refBLK.mesh.faceE.L
        W = interfaceW * refBLK.mesh.faceW.L
        N = interfaceN * refBLK.mesh.faceN.L
        S = interfaceS * refBLK.mesh.faceS.L

        # Compute dUdx
        refBLK.gradx = (E * refBLK.mesh.faceE.xnorm +
                        W * refBLK.mesh.faceW.xnorm +
                        N * refBLK.mesh.faceN.xnorm +
                        S * refBLK.mesh.faceS.xnorm
                        ) / refBLK.mesh.A

        # Compute dUdy
        refBLK.grady = (E * refBLK.mesh.faceE.ynorm +
                        W * refBLK.mesh.faceW.ynorm +
                        N * refBLK.mesh.faceN.ynorm +
                        S * refBLK.mesh.faceS.ynorm
                        ) / refBLK.mesh.A
