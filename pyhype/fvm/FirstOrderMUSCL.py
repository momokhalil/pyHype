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

from typing import TYPE_CHECKING, Union
from pyhype.utils.utils import NumpySlice

if TYPE_CHECKING:
    from pyhype.blocks.quad_block import BaseBlockGhost
    from pyhype.flux.base import FluxFunction
    from pyhype.limiters.base import SlopeLimiter
    from pyhype.gradients.base import Gradient
    from pyhype.solver_config import SolverConfig
    from pyhype.mesh.quadratures import QuadraturePoint

import numpy as np
from pyhype.fvm.base import MUSCL


np.set_printoptions(precision=3)


class FirstOrderMUSCL(MUSCL):
    def __init__(
        self,
        config: SolverConfig,
        flux: FluxFunction,
        limiter: SlopeLimiter,
        gradient: Gradient,
        parent_block: BaseBlockGhost,
    ):
        if config.nghost != 1:
            raise ValueError(
                "Number of ghost cells must be equal to 1 for this method."
            )
        super().__init__(
            config=config,
            flux=flux,
            limiter=limiter,
            gradient=gradient,
            parent_block=parent_block,
        )

    def compute_limiter(self) -> [np.ndarray]:
        """
        No limiting in first order

        :return: None
        """
        return

    def unlimited_solution_at_quadrature_point(
        self,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> [np.ndarray]:
        """
        Returns the cell average values at each quadrature point on all cell faces.

        Parameters:
            - parent_block (QuadBlock): Reference block whose state is to be reconstructed

        Returns:
            - stateE (np.ndarray): Reconstructed values at the east face midpoints of all cells in the block
            - stateW (np.ndarray): Reconstructed values at the west face midpoints of all cells in the block
            - stateN (np.ndarray): Reconstructed values at the north face midpoints of all cells in the block
            - stateS (np.ndarray): Reconstructed values at the south face midpoints of all cells in the block
        """

        # Compute limited values at quadrature points
        stateE = [self.parent_block.state.data.copy() for _ in self.parent_block.qp.E]
        stateW = [self.parent_block.state.data.copy() for _ in self.parent_block.qp.W]
        stateN = [self.parent_block.state.data.copy() for _ in self.parent_block.qp.N]
        stateS = [self.parent_block.state.data.copy() for _ in self.parent_block.qp.S]

        return stateE, stateW, stateN, stateS

    def limited_solution_at_quadrature_point(
        self,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> [np.ndarray]:
        raise RuntimeError("There is no slop elimiting in first order FVM schemes.")
