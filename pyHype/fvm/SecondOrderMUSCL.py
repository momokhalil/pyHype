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
from abc import ABC

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
np.set_printoptions(precision=3)

from typing import TYPE_CHECKING
from pyHype.fvm.base import MUSCLFiniteVolumeMethod

if TYPE_CHECKING:
    from pyHype.mesh.quadratures import QuadraturePoint
    from pyHype.blocks.base import QuadBlock
    from pyHype.states import State


class SecondOrderMUSCL(MUSCLFiniteVolumeMethod):
    def __init__(self, inputs, global_nBLK):
        if inputs.nghost != 1:
            raise ValueError('Number of ghost cells must be equal to 1 for this method.')
        super().__init__(inputs, global_nBLK)

    @staticmethod
    def high_order_term(refBLK: QuadBlock,
                        qp: QuadraturePoint
                        ) -> np.ndarray:
        """
        Compute the high order term used for the state reconstruction at the quadrature point on a specified face.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with solution data and geometry

        :type qp: QuadBlock
        :param qp: Quadrature point data (geometry, weight)

        :rtype: np.ndarray
        :return: Unlimited high order term
        """
        return refBLK.grad.x * (qp.x - refBLK.mesh.x) + refBLK.grad.y * (qp.y - refBLK.mesh.y)

    def unlimited_solution_at_quadrature_point(self,
                                               state: State,
                                               refBLK: QuadBlock,
                                               qp: QuadraturePoint
                                               ) -> np.ndarray:
        """
        Returns the unlimited reconstructed solution at a specific quadrature point based on the given solution state
        and mesh geometry from a reference block.

        :type state: State
        :param state: Solution state

        :type refBLK: QuadBlock
        :param refBLK: Reference block containing mesh geometry data and gradients

        :type qp: QuadBlock
        :param qp: Quadrature point geometry

        :rtype: np.ndarray
        :return: Unlimited reconstructed solution at the quadrature point
        """
        return state + self.high_order_term(refBLK, qp)

    def limited_solution_at_quadrature_point(self,
                                             state: State,
                                             refBLK: QuadBlock,
                                             qp: QuadraturePoint,
                                             ) -> np.ndarray:
        """
        Returns the limited reconstructed solution at a specific quadrature point based on the given solution state and
        slope limiter values and mesh geometry from a reference block.

        :type state: State
        :param state: Solution state

        :type refBLK: QuadBlock
        :param refBLK: Reference block containing mesh geometry data and gradients

        :type qp: QuadBlock
        :param qp: Quadrature point geometry

        :rtype: np.ndarray
        :return: Limited reconstructed solution at the quadrature point
        """
        return state + refBLK.fvm.limiter.phi * self.high_order_term(refBLK, qp)

    def compute_limiter(self, refBLK: QuadBlock) -> None:
        """
        Compute the slope limiter based on the solution data stored in the reconstruction block inside of the given
        reference block. This ensures solution monotonicity when discontinuities exist.

        :type refBLK: QuadBlock
        :param refBLK: Reference block whose state is to be reconstructed

        :rtype: None
        :return: None
        """
        unlimE = [self.unlimited_solution_at_quadrature_point(refBLK.reconBlk.state, refBLK, qp) for qp in refBLK.QP.E]
        unlimW = [self.unlimited_solution_at_quadrature_point(refBLK.reconBlk.state, refBLK, qp) for qp in refBLK.QP.W]
        unlimN = [self.unlimited_solution_at_quadrature_point(refBLK.reconBlk.state, refBLK, qp) for qp in refBLK.QP.N]
        unlimS = [self.unlimited_solution_at_quadrature_point(refBLK.reconBlk.state, refBLK, qp) for qp in refBLK.QP.S]
        self.limiter(refBLK.reconBlk, gqpE=unlimE, gqpW=unlimW, gqpN=unlimN, gqpS=unlimS)

    def integrate_flux_E(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        """
        Integrates the east face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: East face integrated fluxes
        """
        return 0.5 * refBLK.mesh.face.E.L * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.QP.E, self.Flux.E)))

    def integrate_flux_W(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        """
        Integrates the west face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: West face integrated fluxes
        """
        return 0.5 * refBLK.mesh.face.W.L * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.QP.W, self.Flux.W)))

    def integrate_flux_N(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        """
        Integrates the north face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: North face integrated fluxes
        """
        return 0.5 * refBLK.mesh.face.N.L * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.QP.N, self.Flux.N)))

    def integrate_flux_S(self,
                         refBLK: QuadBlock
                         ) -> np.ndarray:
        """
        Integrates the south face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: South face integrated fluxes
        """
        return 0.5 * refBLK.mesh.face.S.L * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.QP.S, self.Flux.S)))
