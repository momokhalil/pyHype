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

from abc import abstractmethod

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput
    from pyHype.blocks.QuadBlock import QuadBlock


class Gradient:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    def __call__(self,
                 refBLK: QuadBlock
                 ) -> None:
        self._get_gradient(refBLK)

    @staticmethod
    @abstractmethod
    def _get_gradient(refBLK: QuadBlock) -> None:
        pass


"""
class LeastSquares9Point:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

        self.stencilSW = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]]
        self.stencilNW = [[-2, 0], [-1, 0], [0, 0], [0, 1], [-2, 0], [-2, 1], [-1, 1]]
        self.stencilSE = [[0, 0], [1, 0], [0, -1], [0, -2], [0, -1], [1, -1], [1, -2]]
        self.stencilNE = [[0, -1], [0, -2], [-1, 0], [-2, 0], [-1, -2], [1, -1], [1, -2]]

    def __call__(self, refBLK):
        return self.least_squares_nearest_neighbor(refBLK)

    def least_squares_nearest_neighbor(self, refBLK: QuadBlock):
        bdr = refBLK.boundary_blocks

        refBLK.grad.x, refBLK.grad.y = least_squares_9_point(refBLK.state.Q,
                                           bdr.E.state.Q, bdr.W.state.Q, bdr.N.state.Q, bdr.S.state.Q,
                                           refBLK.mesh.x, refBLK.mesh.y,
                                           bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                           bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                           self.inputs.nx, self.inputs.ny,
                                           self.stencilSW, self.stencilNW, self.stencilSE, self.stencilNE)
"""


class GreenGauss(Gradient):
    def __init__(self, inputs: ProblemInput):
        super().__init__(inputs)

    @staticmethod
    def _get_gradient(refBLK: QuadBlock) -> None:
        interfaceE, interfaceW, interfaceN, interfaceS = refBLK.reconBlk.get_interface_values()
        # Get each face's contribution to dUdx
        E = interfaceE * refBLK.mesh.face.E.L
        W = interfaceW * refBLK.mesh.face.W.L
        N = interfaceN * refBLK.mesh.face.N.L
        S = interfaceS * refBLK.mesh.face.S.L
        # Compute dUdx
        refBLK.grad.x = (E * refBLK.mesh.face.E.xnorm +
                         W * refBLK.mesh.face.W.xnorm +
                         N * refBLK.mesh.face.N.xnorm +
                         S * refBLK.mesh.face.S.xnorm
                         ) / refBLK.mesh.A
        # Compute dUdy
        refBLK.grad.y = (E * refBLK.mesh.face.E.ynorm +
                         W * refBLK.mesh.face.W.ynorm +
                         N * refBLK.mesh.face.N.ynorm +
                         S * refBLK.mesh.face.S.ynorm
                         ) / refBLK.mesh.A
