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
import numba as nb
from typing import TYPE_CHECKING
from abc import abstractmethod

np.set_printoptions(formatter={"float": "{: 0.3f}".format})


import pyhype.flux as flux
import pyhype.utils.utils as utils
import pyhype.fvm.Gradients as Grads

from pyhype.limiters import limiters
from pyhype.states.primitive import PrimitiveState
from pyhype.states.conservative import ConservativeState
from pyhype.utils.utils import SidePropertyContainer

if TYPE_CHECKING:
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.blocks.base import GradientsContainer
    from pyhype.states.base import State
    from pyhype.states.primitive import PrimitiveState
    from pyhype.mesh.quadratures import QuadraturePoint


class MUSCLFiniteVolumeMethod:
    ALL_IDX = np.s_[:, :, :]

    def __init__(
        self,
        inputs,
    ) -> None:
        """
        Solves the euler equations using a MUSCL-type finite volume scheme.

        TODO:
        ------ DESCRIBE MUSCL BRIEFLY ------

        The matrix structure used for storing solution data in various State classes is a (ny * nx * 4) numpy ndarray
        which has planar dimentions equal to the number of cells in the y and x direction, and a depth of 4. The
        structure looks as follows:

            ___________________nx____________________
            v                                       v
        |>  O----------O----------O----------O----------O ........................ q0 (zeroth state variable)
        |   |          |          |          |          |\
        |   |          |          |          |          |-O ...................... q1 (first state variable)
        |   |          |          |          |          | |\
        |   O----------O----------O----------O----------O |-O .................... q2 (second state variable)
        |   |          |          |          |          |\| |\
        |   |          |          |          |          |-O |-O .................. q3 (third state variable)
        |   |          |          |          |          | |\| |
        ny  O----------O----------O----------O----------O |-O |
        |   |          |          |          |          |\| |\|
        |   |          |          |          |          |-O |-O
        |   |          |          |          |          | |\| |
        |   O----------O----------O----------O----------O |-O |
        |   |          |          |          |          |\| |\|
        |   |          |          |          |          |-O | O
        |   |          |          |          |          | |\| |
        |>  O----------O----------O----------O----------O |-O |
             \|         \|         \|         \|         \| |\|
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O
                          . . .
        ... to be continued.
        """
        self.inputs = inputs

        # Flux storage arrays
        self.Flux = SidePropertyContainer(
            E=tuple(
                np.empty((self.inputs.ny, self.inputs.nx, 4))
                for _ in range(self.inputs.fvm_num_quadrature_points)
            ),
            W=tuple(
                np.empty((self.inputs.ny, self.inputs.nx, 4))
                for _ in range(self.inputs.fvm_num_quadrature_points)
            ),
            N=tuple(
                np.empty((self.inputs.ny, self.inputs.nx, 4))
                for _ in range(self.inputs.fvm_num_quadrature_points)
            ),
            S=tuple(
                np.empty((self.inputs.ny, self.inputs.nx, 4))
                for _ in range(self.inputs.fvm_num_quadrature_points)
            ),
        )
        # Set flux function
        if self.inputs.fvm_flux_function == "Roe":
            self.flux_function_X = flux.FluxRoe(
                self.inputs, size=self.inputs.nx, sweeps=self.inputs.ny
            )
            self.flux_function_Y = flux.FluxRoe(
                self.inputs, size=self.inputs.ny, sweeps=self.inputs.nx
            )
        elif self.inputs.fvm_flux_function == "HLLE":
            self.flux_function_X = flux.FluxHLLE(
                self.inputs, nx=self.inputs.nx, ny=self.inputs.ny
            )
            self.flux_function_Y = flux.FluxHLLE(
                self.inputs, nx=self.inputs.ny, ny=self.inputs.nx
            )
        elif self.inputs.fvm_flux_function == "HLLL":
            self.flux_function_X = flux.FluxHLLL(
                self.inputs, nx=self.inputs.nx, ny=self.inputs.ny
            )
            self.flux_function_Y = flux.FluxHLLL(
                self.inputs, nx=self.inputs.ny, ny=self.inputs.nx
            )
        else:
            raise ValueError(
                "MUSCLFiniteVolumeMethod: Flux function type not specified."
            )

        # Set slope limiter
        if self.inputs.fvm_slope_limiter == "VanLeer":
            self.limiter = limiters.VanLeer(self.inputs)
        elif self.inputs.fvm_slope_limiter == "VanAlbada":
            self.limiter = limiters.VanAlbada(self.inputs)
        elif self.inputs.fvm_slope_limiter == "Venkatakrishnan":
            self.limiter = limiters.Venkatakrishnan(self.inputs)
        elif self.inputs.fvm_slope_limiter == "BarthJespersen":
            self.limiter = limiters.BarthJespersen(self.inputs)
        else:
            raise ValueError(
                "MUSCLFiniteVolumeMethod: Slope limiter type not specified."
            )

        # Set gradient algorithm
        if self.inputs.fvm_gradient_type == "GreenGauss":
            self.gradient = Grads.GreenGauss(self.inputs)
        else:
            raise ValueError(
                "MUSCLFiniteVolumeMethod: Slope limiter type not specified."
            )

    def basis_state(self):
        if self.inputs.reconstruction_type == "primitive":
            return PrimitiveState
        if self.inputs.reconstruction_type == "conservative":
            return ConservativeState

    def reconstruct(self, refBLK: QuadBlock) -> None:
        """
        This method performs the steps needed to complete the spatial reconstruction of the solution state, which is
        part of the spatial discretization required to solve the finite volume problem. The reconstruction process has
        three key components:
          1) Transformation of the state solution into the correct reconstruction basis.
          2) Computation of the gradients
          3) Computation of the slope limiter to ensure monotonicity

        :type refBLK: QuadBlock
        :param refBLK: Reference block to reconstruct

        :rtype: None
        :return: None
        """
        refBLK.reconBlk.from_block(refBLK)
        self.gradient(refBLK)
        self.compute_limiter(refBLK)

    @abstractmethod
    def compute_limiter(self, refBLK: QuadBlock) -> [np.ndarray]:
        """
        Implementation of the reconstruction method specialized to the Finite Volume Method described in the class.
        """
        raise NotImplementedError

    def integrate_flux_E(self, refBLK: QuadBlock) -> np.ndarray:
        """
        Integrates the east face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: East face integrated fluxes
        """
        return (
            0.5
            * refBLK.mesh.face.E.L
            * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.qp.E, self.Flux.E)))
        )

    def integrate_flux_W(self, refBLK: QuadBlock) -> np.ndarray:
        """
        Integrates the west face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: West face integrated fluxes
        """
        return (
            0.5
            * refBLK.mesh.face.W.L
            * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.qp.W, self.Flux.W)))
        )

    def integrate_flux_N(self, refBLK: QuadBlock) -> np.ndarray:
        """
        Integrates the north face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: North face integrated fluxes
        """
        return (
            0.5
            * refBLK.mesh.face.N.L
            * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.qp.N, self.Flux.N)))
        )

    def integrate_flux_S(self, refBLK: QuadBlock) -> np.ndarray:
        """
        Integrates the south face fluxes using an n-point gauss gradrature rule.

        :type refBLK: QuadBlock
        :param refBLK: Reference block with flux data for integration

        :rtype: np.ndarray
        :return: South face integrated fluxes
        """
        return (
            0.5
            * refBLK.mesh.face.S.L
            * sum((qp.w * qpflux for (qp, qpflux) in zip(refBLK.qp.S, self.Flux.S)))
        )

    @staticmethod
    @abstractmethod
    def high_order_term(
        refBLK: QuadBlock, qp: QuadraturePoint, slicer: slice or tuple or int = None
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def limited_solution_at_quadrature_point(
        self,
        state: State,
        gradients: GradientsContainer,
        qp: QuadraturePoint,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def unlimited_solution_at_quadrature_point(
        self,
        state: State,
        gradients: GradientsContainer,
        qp: QuadraturePoint,
        slicer: slice or tuple or int = None,
    ) -> np.ndarray:
        raise NotImplementedError

    def dUdt(self, refBLK: QuadBlock):
        """
        Compute residuals used for marching the solution through time by integrating the fluxes on each cell face and
        applying the semi-discrete Godunov method:

        dUdt[i] = - (1/A[i]) * sum[over all faces] (F[face] * length[face])
        """
        self.evaluate_flux(refBLK)
        fluxE = self.integrate_flux_E(refBLK)
        fluxW = self.integrate_flux_W(refBLK)
        fluxN = self.integrate_flux_N(refBLK)
        fluxS = self.integrate_flux_S(refBLK)
        if self.inputs.use_JIT:
            return self._dUdt_JIT(fluxE, fluxW, fluxN, fluxS, refBLK.mesh.A)
        return (fluxW - fluxE + fluxS - fluxN) / refBLK.mesh.A

    @staticmethod
    @nb.njit(cache=True)
    def _dUdt_JIT(fluxE, fluxW, fluxN, fluxS, A):
        dUdt = np.zeros_like(fluxE)
        for i in range(fluxE.shape[0]):
            for j in range(fluxE.shape[1]):
                a = A[i, j, 0]
                for k in range(fluxE.shape[2]):
                    dUdt[i, j, k] = (
                        fluxW[i, j, k]
                        - fluxE[i, j, k]
                        + fluxS[i, j, k]
                        - fluxN[i, j, k]
                    ) / a
        return dUdt

    def get_LR_states_for_EW_fluxes(
        self,
        ghostE: State,
        ghostW: State,
        stateE: State,
        stateW: State,
    ) -> [PrimitiveState]:
        """
        Compute and return the left and right states used for the East-West inviscid flux calculation. The left state
        is created by concatenating the west ghost-cell state and the east face state. The right state
        is created by concatenating the east ghost-cell state and the west face state. After concatenation, the arrays
        are reshaped to produce a (1, n, 4) shaped array.

        :type ghostE: np.ndarray
        :param ghostE: east ghost cell state array

        :type ghostW: np.ndarray
        :param ghostW: west ghost cell state array

        :type stateE: np.ndarray
        :param stateE: east state array

        :type stateW: np.ndarray
        :param stateW: west state array

        :rtype: tuple(PrimitiveState, PrimitiveState)
        :return: PrimitiveStates that hold the left and right states for the flux calculation
        """
        _arrL = np.concatenate((ghostW.Q, stateE.Q), axis=1)
        _arrR = np.concatenate((stateW.Q, ghostE.Q), axis=1)

        _L = self.basis_state()(self.inputs, array=_arrL)
        _R = self.basis_state()(self.inputs, array=_arrR)

        _stateL = PrimitiveState(inputs=self.inputs, state=_L)
        _stateR = PrimitiveState(inputs=self.inputs, state=_R)
        return _stateL, _stateR

    def get_LR_states_for_NS_fluxes(
        self,
        ghostN: State,
        ghostS: State,
        stateN: State,
        stateS: State,
    ) -> [PrimitiveState]:
        """
        Compute and return the left and right states used for the North-South inviscid flux calculation. The left state
        is created by concatenating the south ghost-cell state and the north face state. The right state
        is created by concatenating the north ghost-cell state and the south face state. After concatenation, the arrays
        are reshaped to produce a (1, n, 4) shaped array.

        :type ghostN: np.ndarray
        :param ghostN: north ghost cell state array

        :type ghostS: np.ndarray
        :param ghostS: south ghost cell state array

        :type stateN: np.ndarray
        :param stateN: north state array

        :type stateS: np.ndarray
        :param stateS: south state array

        :rtype: tuple(PrimitiveState, PrimitiveState)
        :return: PrimitiveStates that hold the left and right states for the flux calculation
        """
        _arrL = np.concatenate((ghostS.Q, stateN.Q), axis=0).transpose((1, 0, 2))
        _arrR = np.concatenate((stateS.Q, ghostN.Q), axis=0).transpose((1, 0, 2))

        _L = self.basis_state()(self.inputs, array=_arrL)
        _R = self.basis_state()(self.inputs, array=_arrR)

        _stateL = PrimitiveState(inputs=self.inputs, state=_L)
        _stateR = PrimitiveState(inputs=self.inputs, state=_R)

        return _stateL, _stateR

    def evaluate_flux_EW(self, refBLK: QuadBlock) -> None:
        """
        Evaluates the fluxes at each east-west cell boundary. The following steps are followed:
            1. Get list of reconstructed boundary states at each quadrature point on the east boundary
            2. Get list of reconstructed boundary states at each quadrature point on the west boundary
            3. Loop through each quadrature point on the east and west faces and calculate the reconstructed solution
               states at each q-point
            4. Rotate states if the block is not cartesian
            5. Compute left and right states
            6. Compute fluxes

        :type refBLK: QuadBlock
        :param refBLK: Block that holds solution state

        :rtype: None
        :return: None
        """
        bndE = (
            refBLK.reconBlk.get_east_boundary_states_at_qp()
            if refBLK.ghost.E.BCtype != "None"
            else refBLK.reconBlk.ghost.E.get_west_boundary_states_at_qp()
        )
        bndW = (
            refBLK.reconBlk.get_west_boundary_states_at_qp()
            if refBLK.ghost.W.BCtype != "None"
            else refBLK.reconBlk.ghost.W.get_east_boundary_states_at_qp()
        )

        for qe, qw, _bndE, _bndW, fluxE, fluxW in zip(
            refBLK.qp.E, refBLK.qp.W, bndE, bndW, self.Flux.E, self.Flux.W
        ):
            _stateE = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK.reconBlk.state, refBLK, qe
            )
            _stateW = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK.reconBlk.state, refBLK, qw
            )
            refBLK.ghost.E.set_BC(_bndE, state_type=self.inputs.reconstruction_type)
            refBLK.ghost.W.set_BC(_bndW, state_type=self.inputs.reconstruction_type)

            if not refBLK.is_cartesian:
                utils.rotate(refBLK.mesh.face.E.theta, _stateE.Q)
                utils.rotate(refBLK.mesh.face.W.theta, _stateW.Q)
                utils.rotate(refBLK.mesh.east_boundary_angle(), _bndE.Q)
                utils.rotate(refBLK.mesh.west_boundary_angle(), _bndW.Q)

            sL, sR = self.get_LR_states_for_EW_fluxes(_bndE, _bndW, _stateE, _stateW)
            fluxEW = self.flux_function_X(WL=sL, WR=sR)
            fluxE[:] = fluxEW[:, 1:, :]
            fluxW[:] = fluxEW[:, :-1, :]

            if not refBLK.is_cartesian:
                utils.unrotate(refBLK.mesh.face.E.theta, fluxE.Q)
                utils.unrotate(refBLK.mesh.face.W.theta, fluxW.Q)

    def evaluate_flux_NS(self, refBLK: QuadBlock) -> None:
        """
        Evaluates the fluxes at each north-south cell boundary. The following steps are followed:
            1. Get list of reconstructed boundary states at each quadrature point on the north boundary
            2. Get list of reconstructed boundary states at each quadrature point on the south boundary
            3. Loop through each quadrature point on the north and south faces and calculate the reconstructed solution
               states at each q-point
            4. Rotate states if the block is not cartesian
            5. Compute left and right states
            6. Compute fluxes

        :type refBLK: QuadBlock
        :param refBLK: Block that holds solution state

        :rtype: None
        :return: None
        """
        bndN = (
            refBLK.reconBlk.get_north_boundary_states_at_qp()
            if refBLK.ghost.N.BCtype != "None"
            else refBLK.reconBlk.ghost.N.get_south_boundary_states_at_qp()
        )
        bndS = (
            refBLK.reconBlk.get_south_boundary_states_at_qp()
            if refBLK.ghost.S.BCtype != "None"
            else refBLK.reconBlk.ghost.S.get_north_boundary_states_at_qp()
        )

        for qn, qs, _bndN, _bndS, fluxN, fluxS in zip(
            refBLK.qp.N, refBLK.qp.S, bndN, bndS, self.Flux.N, self.Flux.S
        ):
            _stateN = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK.reconBlk.state, refBLK, qn
            )
            _stateS = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK.reconBlk.state, refBLK, qs
            )
            refBLK.ghost.N.set_BC(_bndN, state_type=self.inputs.reconstruction_type)
            refBLK.ghost.S.set_BC(_bndS, state_type=self.inputs.reconstruction_type)

            if refBLK.is_cartesian:
                utils.rotate90(_stateN.Q, _stateS.Q, _bndN.Q, _bndS.Q)
            else:
                utils.rotate(refBLK.mesh.face.N.theta, _stateN.Q)
                utils.rotate(refBLK.mesh.face.S.theta, _stateS.Q)
                utils.rotate(refBLK.mesh.north_boundary_angle(), _bndN.Q)
                utils.rotate(refBLK.mesh.south_boundary_angle(), _bndS.Q)

            sL, sR = self.get_LR_states_for_NS_fluxes(_bndN, _bndS, _stateN, _stateS)
            fluxNS = self.flux_function_Y(WL=sL, WR=sR).transpose((1, 0, 2))
            fluxN[:] = fluxNS[1:, :, :]
            fluxS[:] = fluxNS[:-1, :, :]

            if refBLK.is_cartesian:
                utils.unrotate90(fluxN, fluxS)
            else:
                utils.unrotate(refBLK.mesh.face.N.theta, fluxN.Q)
                utils.unrotate(refBLK.mesh.face.S.theta, fluxS.Q)

    def evaluate_flux(self, refBLK: QuadBlock) -> None:
        """
        Calculates the fluxes at all cell boundaries. Solves the 1-D riemann problem along all of the rows and columns
        of cells on the blocks in a sweeping (but unsplit) fashion.

        :type refBLK: QuadBlock
        :param refBLK: QuadBlock that holds the solution data for the flux calculation

        :rtype: None
        :return: None
        """

        self.reconstruct(refBLK)
        self.evaluate_flux_EW(refBLK)
        self.evaluate_flux_NS(refBLK)
