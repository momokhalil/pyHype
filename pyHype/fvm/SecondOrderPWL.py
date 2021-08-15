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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyHype.blocks.base import QuadBlock
    from pyHype.mesh.base import CellFace

import numpy as np
from pyHype.fvm.base import MUSCLFiniteVolumeMethod


np.set_printoptions(precision=3)


class SecondOrderPWL(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        if inputs.nghost != 1:
            raise ValueError('Number of ghost cells must be equal to 1 for this method.')
        else:
            super().__init__(inputs, global_nBLK)

    @staticmethod
    def high_order_term(refBLK: QuadBlock,
                        face: CellFace):
        """
        Compute the high order term used for the state reconstruction at the quadrature point on a specified face.

        Parameters:
            - refBLK (QuadBlock): QuadBlock class that stores all data related to the block of interest
            - face (Cellface): CellFace class that stores the mesh data for the face of interest

        Returns:
            - high_ord (np.ndarray): High order term
        """

        return refBLK.gradx * (face.xmid - refBLK.mesh.x) + refBLK.grady * (face.ymid - refBLK.mesh.y)

    def reconstruct_state(self,
                          refBLK: QuadBlock
                          ) -> [np.ndarray]:

        # High order terms for each cell face
        high_ord_E = self.high_order_term(refBLK, refBLK.mesh.faceE)
        high_ord_W = self.high_order_term(refBLK, refBLK.mesh.faceW)
        high_ord_N = self.high_order_term(refBLK, refBLK.mesh.faceN)
        high_ord_S = self.high_order_term(refBLK, refBLK.mesh.faceS)

        # Compute slope limiter
        phi = self.flux_limiter.limit(refBLK,
                                      quadE=refBLK.state + high_ord_E,
                                      quadW=refBLK.state + high_ord_W,
                                      quadN=refBLK.state + high_ord_N,
                                      quadS=refBLK.state + high_ord_S)

        # Compute limited values at quadrature points
        stateE = refBLK.state + phi * high_ord_E
        stateW = refBLK.state + phi * high_ord_W
        stateN = refBLK.state + phi * high_ord_N
        stateS = refBLK.state + phi * high_ord_S

        return stateE, stateW, stateN, stateS

    def integrate_flux_E(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        return self.Flux_E * refBLK.mesh.faceE.L

    def integrate_flux_W(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        return -self.Flux_W * refBLK.mesh.faceW.L

    def integrate_flux_N(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        return self.Flux_N * refBLK.mesh.faceN.L

    def integrate_flux_S(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        return -self.Flux_S * refBLK.mesh.faceS.L


