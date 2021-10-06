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


class FirstOrder(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):

        if inputs.nghost != 1:
            raise ValueError('Number of ghost cells must be equal to 1 for this method.')
        else:
            super().__init__(inputs, global_nBLK)

    def reconstruct_state(self,
                          refBLK: QuadBlock
                          ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the cell average values at each quadrature point on all cell faces.

        Parameters:
            - refBLK (QuadBlock): Reference block whose state is to be reconstructed

        Returns:
            - stateE (np.ndarray): Reconstructed values at the east face midpoints of all cells in the block
            - stateW (np.ndarray): Reconstructed values at the west face midpoints of all cells in the block
            - stateN (np.ndarray): Reconstructed values at the north face midpoints of all cells in the block
            - stateS (np.ndarray): Reconstructed values at the south face midpoints of all cells in the block
        """

        # Compute limited values at quadrature points
        stateE = refBLK.state.U.copy()
        stateW = refBLK.state.U.copy()
        stateN = refBLK.state.U.copy()
        stateS = refBLK.state.U.copy()

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
