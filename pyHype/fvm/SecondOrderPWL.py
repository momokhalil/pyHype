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
    def high_order_E(refBLK: QuadBlock):

        if refBLK.reconstruction_type == 'primitive':
            return refBLK.dWdx * (refBLK.mesh.faceE.xmid - refBLK.mesh.x) \
                 + refBLK.dWdy * (refBLK.mesh.faceE.ymid - refBLK.mesh.y)
        elif refBLK.reconstruction_type == 'conservative':
            return refBLK.dUdx * (refBLK.mesh.faceE.xmid - refBLK.mesh.x) \
                 + refBLK.dUdy * (refBLK.mesh.faceE.ymid - refBLK.mesh.y)
        else:
            raise ValueError('Reconstruction type ' + str(refBLK.reconstruction_type) + ' is not defined.')

    @staticmethod
    def high_order_W(refBLK: QuadBlock):

        if refBLK.reconstruction_type == 'primitive':
            return refBLK.dWdx * (refBLK.mesh.faceW.xmid - refBLK.mesh.x) \
                 + refBLK.dWdy * (refBLK.mesh.faceW.ymid - refBLK.mesh.y)
        elif refBLK.reconstruction_type == 'conservative':
            return refBLK.dUdx * (refBLK.mesh.faceW.xmid - refBLK.mesh.x) \
                 + refBLK.dUdy * (refBLK.mesh.faceW.ymid - refBLK.mesh.y)
        else:
            raise ValueError('Reconstruction type ' + str(refBLK.reconstruction_type) + ' is not defined.')

    @staticmethod
    def high_order_N(refBLK: QuadBlock):

        if refBLK.reconstruction_type == 'primitive':
            return refBLK.dWdx * (refBLK.mesh.faceN.xmid - refBLK.mesh.x) \
                 + refBLK.dWdy * (refBLK.mesh.faceN.ymid - refBLK.mesh.y)
        elif refBLK.reconstruction_type == 'conservative':
            return refBLK.dUdx * (refBLK.mesh.faceN.xmid - refBLK.mesh.x) \
                 + refBLK.dUdy * (refBLK.mesh.faceN.ymid - refBLK.mesh.y)
        else:
            raise ValueError('Reconstruction type ' + str(refBLK.reconstruction_type) + ' is not defined.')

    @staticmethod
    def high_order_S(refBLK: QuadBlock):

        if refBLK.reconstruction_type == 'primitive':
            return refBLK.dWdx * (refBLK.mesh.faceS.xmid - refBLK.mesh.x) \
                 + refBLK.dWdy * (refBLK.mesh.faceS.ymid - refBLK.mesh.y)
        elif refBLK.reconstruction_type == 'conservative':
            return refBLK.dUdx * (refBLK.mesh.faceS.xmid - refBLK.mesh.x) \
                 + refBLK.dUdy * (refBLK.mesh.faceS.ymid - refBLK.mesh.y)
        else:
            raise ValueError('Reconstruction type ' + str(refBLK.reconstruction_type) + ' is not defined.')

    def reconstruct_state(self,
                          refBLK: QuadBlock,
                          state: np.ndarray,
                          ghostE: np.ndarray,
                          ghostW: np.ndarray,
                          ghostN: np.ndarray,
                          ghostS: np.ndarray
                          ) -> [np.ndarray]:

        # High order terms for each cell face
        high_ord_E = self.high_order_E(refBLK)
        high_ord_W = self.high_order_W(refBLK)
        high_ord_N = self.high_order_N(refBLK)
        high_ord_S = self.high_order_S(refBLK)

        # Compute slope limiter
        phi = self.flux_limiter.limit(state,
                                      ghostE, ghostW, ghostN, ghostS,
                                      quadE=state + high_ord_E,
                                      quadW=state + high_ord_W,
                                      quadN=state + high_ord_N,
                                      quadS=state + high_ord_S)

        # Compute limited values at quadrature points
        stateE = state + phi * high_ord_E
        stateW = state + phi * high_ord_W
        stateN = state + phi * high_ord_N
        stateS = state + phi * high_ord_S

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


