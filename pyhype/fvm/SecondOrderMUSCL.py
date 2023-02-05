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

import numpy as np

np.set_printoptions(precision=3)

from typing import TYPE_CHECKING, Union
from pyhype.fvm.base import MUSCL
from pyhype.utils.utils import NumpySlice

if TYPE_CHECKING:
    from pyhype.mesh.quadratures import QuadraturePoint
    from pyhype.blocks.base import QuadBlock, BaseBlockFVM
    from pyhype.states import State
    from pyhype.flux.base import FluxFunction
    from pyhype.limiters.base import SlopeLimiter
    from pyhype.gradients.base import Gradient
    from pyhype.solver_config import SolverConfig


class SecondOrderMUSCL(MUSCL):
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
        super().__init__(config=config, limiter=limiter, flux=flux, gradient=gradient)

    @staticmethod
    def high_order_term(
        parent_block: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> np.ndarray:
        """
        Compute the high order term used for the state reconstruction at the quadrature point on a specified face.

        :type parent_block: QuadBlock
        :param parent_block: Reference block with solution data and geometry

        :type qp: QuadBlock
        :param qp: Quadrature point data (geometry, weight)

        :type slicer: slice
        :param slicer: Numpy array slice object (which is actually a tuple)

        :rtype: np.ndarray
        :return: Unlimited high order term
        """
        return parent_block.grad.get_high_order_term(
            parent_block.mesh.x, qp.x, parent_block.mesh.y, qp.y, slicer=slicer
        )

    def unlimited_solution_at_quadrature_point(
        self,
        parent_block: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> np.ndarray:
        """
        Returns the unlimited reconstructed solution at a specific quadrature point based on the given solution state
        and mesh geometry from a reference block.

        :type parent_block: QuadBlock
        :param parent_block: Reference block containing mesh geometry data and gradients

        :type qp: QuadBlock
        :param qp: Quadrature point geometry

        :type slicer: slice or int or tuple
        :param slicer: Numpy array slice object (which is actually a tuple)

        :rtype: np.ndarray
        :return: Unlimited reconstructed solution at the quadrature point
        """
        return parent_block.state[slicer] + self.high_order_term(
            parent_block, qp, slicer
        )

    def limited_solution_at_quadrature_point(
        self,
        parent_block: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> np.ndarray:
        """
        Returns the limited reconstructed solution at a specific quadrature point based on the given solution state and
        slope limiter values and mesh geometry from a reference block.

        :type parent_block: QuadBlock
        :param parent_block: Reference block containing mesh geometry data and gradients

        :type qp: QuadBlock
        :param qp: Quadrature point geometry

        :type slicer: slice or int or tuple
        :param slicer: Numpy array slice object (which is actually a tuple)

        :rtype: np.ndarray
        :return: Limited reconstructed solution at the quadrature point
        """
        return parent_block.state[slicer] + self.limiter.phi[
            slicer
        ] * self.high_order_term(parent_block, qp, slicer)

    def compute_limiter(self, parent_block: QuadBlock) -> None:
        """
        Compute the slope limiter based on the solution data stored in the reconstruction block inside of the given
        reference block. This ensures solution monotonicity when discontinuities exist.

        :type parent_block: QuadBlock
        :param parent_block: Reference block whose state is to be reconstructed

        :rtype: None
        :return: None
        """
        east_unlimited_states = [
            self.unlimited_solution_at_quadrature_point(
                parent_block=parent_block, qp=qp
            )
            for qp in parent_block.qp.E
        ]
        west_unlimited_states = [
            self.unlimited_solution_at_quadrature_point(
                parent_block=parent_block, qp=qp
            )
            for qp in parent_block.qp.W
        ]
        north_unlimited_states = [
            self.unlimited_solution_at_quadrature_point(
                parent_block=parent_block, qp=qp
            )
            for qp in parent_block.qp.N
        ]
        south_unlimited_states = [
            self.unlimited_solution_at_quadrature_point(
                parent_block=parent_block, qp=qp
            )
            for qp in parent_block.qp.S
        ]
        self.limiter.limit(
            parent_block,
            gqpE=east_unlimited_states,
            gqpW=west_unlimited_states,
            gqpN=north_unlimited_states,
            gqpS=south_unlimited_states,
        )
