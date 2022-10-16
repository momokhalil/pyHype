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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.blocks.base import QuadBlock
    from pyhype.mesh.base import CellFace

import numpy as np
from pyhype.fvm.base import MUSCLFiniteVolumeMethod


np.set_printoptions(precision=3)


class FirstOrderMUSCL(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs):
        if inputs.nghost != 1:
            raise ValueError(
                "Number of ghost cells must be equal to 1 for this method."
            )
        super().__init__(inputs)

    def reconstruct_state(
        self, refBLK: QuadBlock
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
        _stateE = refBLK.state.data.copy()
        stateE = [_stateE for _ in refBLK.qp.E]

        _stateW = refBLK.state.data.copy()
        stateW = [_stateW for _ in refBLK.qp.W]

        _stateN = refBLK.state.data.copy()
        stateN = [_stateN for _ in refBLK.qp.N]

        _stateS = refBLK.state.data.copy()
        stateS = [_stateS for _ in refBLK.qp.S]

        return stateE, stateW, stateN, stateS