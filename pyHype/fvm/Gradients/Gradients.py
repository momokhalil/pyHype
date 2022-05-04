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
import numba as nb
import numpy as np

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
        """
        Interface to call the gradient algorithm.

        :type refBLK: QuadBlock
        :param refBLK: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        self._get_gradient(refBLK)

    @staticmethod
    @abstractmethod
    def _get_gradient(refBLK: QuadBlock) -> None:
        """
        Implementation of the gradient algorithm.

        :type refBLK: QuadBlock
        :param refBLK: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        raise NotImplementedError


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
    @staticmethod
    def _get_gradient_NUMPY(refBLK: QuadBlock) -> None:
        """
        Compute the x and y direction gradients using the Green-Gauss method, implemeted fully in numpy.

        :type refBLK: QuadBlock
        :param refBLK: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        face = refBLK.mesh.face
        interfaceE, interfaceW, interfaceN, interfaceS = refBLK.reconBlk.get_interface_values()

        # Get each face's contribution to dUdx
        E = interfaceE * face.E.L
        W = interfaceW * face.W.L
        N = interfaceN * face.N.L
        S = interfaceS * face.S.L

        # Compute dUdx
        refBLK.grad.x = (E * face.E.xnorm + W * face.W.xnorm + N * face.N.xnorm + S * face.S.xnorm) / refBLK.mesh.A
        # Compute dUdy
        refBLK.grad.y = (E * face.E.ynorm + W * face.W.ynorm + N * face.N.ynorm + S * face.S.ynorm) / refBLK.mesh.A

    def _get_gradient(self, refBLK: QuadBlock) -> None:
        """
        Compute the x and y direction gradients using the Green-Gauss method, implemented with numba JIT.

        :type refBLK: QuadBlock
        :param refBLK: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        interfaceE, interfaceW, interfaceN, interfaceS = refBLK.reconBlk.get_interface_values()
        self._get_gradinet_JIT(interfaceE * refBLK.mesh.face.E.L,
                               interfaceW * refBLK.mesh.face.W.L,
                               interfaceN * refBLK.mesh.face.N.L,
                               interfaceS * refBLK.mesh.face.S.L,
                               refBLK.mesh.face.E.xnorm,
                               refBLK.mesh.face.W.xnorm,
                               refBLK.mesh.face.N.xnorm,
                               refBLK.mesh.face.S.xnorm,
                               refBLK.mesh.face.E.ynorm,
                               refBLK.mesh.face.W.ynorm,
                               refBLK.mesh.face.N.ynorm,
                               refBLK.mesh.face.S.ynorm,
                               refBLK.mesh.A,
                               refBLK.grad.x,
                               refBLK.grad.y)

    @staticmethod
    @nb.njit(cache=True)
    def _get_gradinet_JIT(E: np.ndarray, W: np.ndarray, N: np.ndarray, S: np.ndarray,
                          xE: np.ndarray, xW: np.ndarray, xN: np.ndarray, xS: np.ndarray,
                          yE: np.ndarray, yW: np.ndarray, yN: np.ndarray, yS: np.ndarray,
                          A: np.ndarray, gx: np.ndarray, gy: np.ndarray):
        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                for k in range(E.shape[2]):
                    a = 1 / A[i, j, 0]
                    gx[i, j, k] = (E[i, j, k] * xE[i, j, 0] +
                                   W[i, j, k] * xW[i, j, 0] +
                                   N[i, j, k] * xN[i, j, 0] +
                                   S[i, j, k] * xS[i, j, 0]
                                   ) * a
                    gy[i, j, k] = (E[i, j, k] * yE[i, j, 0] +
                                   W[i, j, k] * yW[i, j, 0] +
                                   N[i, j, k] * yN[i, j, 0] +
                                   S[i, j, k] * yS[i, j, 0]
                                   ) * a
