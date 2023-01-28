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
    from pyhype.states.base import State
    from pyhype.blocks.base import QuadBlock
    from pyhype.flux.base import FluxFunction
    from pyhype.blocks.base import BaseBlockFVM
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
    ):
        if config.nghost != 1:
            raise ValueError(
                "Number of ghost cells must be equal to 1 for this method."
            )
        super().__init__(config, limiter, flux, gradient)

    def compute_limiter(self, refBLK: QuadBlock) -> [np.ndarray]:
        """
        No limiting in first order

        :param refBLK: Parent block
        :return: None
        """
        pass

    def unlimited_solution_at_quadrature_point(
        self,
        refBLK: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> [np.ndarray]:
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
        stateE = [refBLK.state.data.copy() for _ in refBLK.qp.E]
        stateW = [refBLK.state.data.copy() for _ in refBLK.qp.W]
        stateN = [refBLK.state.data.copy() for _ in refBLK.qp.N]
        stateS = [refBLK.state.data.copy() for _ in refBLK.qp.S]

        return stateE, stateW, stateN, stateS

    def limited_solution_at_quadrature_point(
        self,
        refBLK: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> [np.ndarray]:
        """
        No limiting or high order terms in a first order MUSCL method,
        simply return the unlimited solution at the quadrature point.

        :type refBLK: QuadBlock
        :param refBLK: Reference block containing mesh geometry data and gradients

        :type qp: QuadBlock
        :param qp: Quadrature point geometry

        :type slicer: slice or int or tuple
        :param slicer: Numpy array slice object (which is actually a tuple)

        :rtype: [np.ndarray]
        :return: Unlimited solution at the quadrature point
        """
        return self.unlimited_solution_at_quadrature_point(
            refBLK=refBLK,
            qp=qp,
            slicer=slicer,
        )
