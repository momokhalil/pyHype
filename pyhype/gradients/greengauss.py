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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
from abc import abstractmethod
from pyhype.gradients.base import Gradient

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.solvers.base import SolverConfig
    from pyhype.blocks.quad_block import QuadBlock


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
        (
            interfaceE,
            interfaceW,
            interfaceN,
            interfaceS,
        ) = refBLK.reconBlk.get_interface_values()

        # Get each face's contribution to dUdx
        E = interfaceE * face.E.L
        W = interfaceW * face.W.L
        N = interfaceN * face.N.L
        S = interfaceS * face.S.L

        # Compute dUdx
        refBLK.grad.x = (
            E * face.E.norm.x
            + W * face.W.norm.x
            + N * face.N.norm.x
            + S * face.S.norm.x
        ) / refBLK.mesh.A
        # Compute dUdy
        refBLK.grad.y = (
            E * face.E.norm.y
            + W * face.W.norm.y
            + N * face.N.norm.y
            + S * face.S.norm.y
        ) / refBLK.mesh.A

    def _get_gradient(self, refBLK: QuadBlock) -> None:
        """
        Compute the x and y direction gradients using the Green-Gauss method, implemented with numba JIT.

        :type refBLK: QuadBlock
        :param refBLK: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        (
            interfaceE,
            interfaceW,
            interfaceN,
            interfaceS,
        ) = refBLK.reconBlk.get_interface_values()
        self._get_gradinet_JIT(
            interfaceE,
            interfaceW,
            interfaceN,
            interfaceS,
            refBLK.mesh.face.E.L[:, :, 0],
            refBLK.mesh.face.W.L[:, :, 0],
            refBLK.mesh.face.N.L[:, :, 0],
            refBLK.mesh.face.S.L[:, :, 0],
            refBLK.mesh.face.E.norm.x[:, :, 0],
            refBLK.mesh.face.W.norm.x[:, :, 0],
            refBLK.mesh.face.N.norm.x[:, :, 0],
            refBLK.mesh.face.S.norm.x[:, :, 0],
            refBLK.mesh.face.E.norm.y[:, :, 0],
            refBLK.mesh.face.W.norm.y[:, :, 0],
            refBLK.mesh.face.N.norm.y[:, :, 0],
            refBLK.mesh.face.S.norm.y[:, :, 0],
            refBLK.mesh.A[:, :, 0],
            refBLK.grad.x,
            refBLK.grad.y,
        )

    @staticmethod
    @nb.njit(cache=True)
    def _get_gradinet_JIT(
        E: np.ndarray,
        W: np.ndarray,
        N: np.ndarray,
        S: np.ndarray,
        lE: np.ndarray,
        lW: np.ndarray,
        lN: np.ndarray,
        lS: np.ndarray,
        xE: np.ndarray,
        xW: np.ndarray,
        xN: np.ndarray,
        xS: np.ndarray,
        yE: np.ndarray,
        yW: np.ndarray,
        yN: np.ndarray,
        yS: np.ndarray,
        A: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
    ):

        for i in range(E.shape[0]):
            for j in range(E.shape[1]):
                _lE = lE[i, j]
                _lW = lW[i, j]
                _lN = lN[i, j]
                _lS = lS[i, j]
                xlE = _lE * xE[i, j]
                xlW = _lW * xW[i, j]
                xlN = _lN * xN[i, j]
                xlS = _lS * xS[i, j]
                ylE = _lE * yE[i, j]
                ylW = _lW * yW[i, j]
                ylN = _lN * yN[i, j]
                ylS = _lS * yS[i, j]
                a = 1.0 / A[i, j]
                for k in range(E.shape[2]):
                    e = E[i, j, k]
                    w = W[i, j, k]
                    n = N[i, j, k]
                    s = S[i, j, k]
                    gx[i, j, k] = (e * xlE + w * xlW + n * xlN + s * xlS) * a
                    gy[i, j, k] = (e * ylE + w * ylW + n * ylN + s * ylS) * a
