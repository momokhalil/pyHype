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

from abc import abstractmethod, ABC
from typing import Union, TYPE_CHECKING
import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numba as nb
import numpy as np

np.set_printoptions(formatter={"float": "{: 0.3f}".format})

import pyhype.utils.utils as utils
from pyhype.utils.utils import NumpySlice
from pyhype.utils.utils import SidePropertyContainer
from pyhype.states.primitive import PrimitiveState

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.blocks.base import BaseBlockFVM
    from pyhype.mesh.quadratures import QuadraturePoint
    from pyhype.flux.base import FluxFunction
    from pyhype.limiters.base import SlopeLimiter
    from pyhype.gradients.base import Gradient


class FiniteVolumeMethod(ABC):
    east_face_slice = NumpySlice.east_face()
    west_face_slice = NumpySlice.west_face()
    north_face_slice = NumpySlice.north_face()
    south_face_slice = NumpySlice.south_face()
    east_boundary_slice = NumpySlice.east_boundary()
    west_boundary_slice = NumpySlice.west_boundary()
    north_boundary_slice = NumpySlice.north_boundary()
    south_boundary_slice = NumpySlice.south_boundary()

    def __init__(
        self,
        config,
        flux: [FluxFunction],
    ):
        self.config = config
        self.flux_function_x, self.flux_function_y = flux

    @abstractmethod
    def compute_limiter(self, refBLK: QuadBlock) -> [np.ndarray]:
        """
        Implementation of the reconstruction method specialized to the Finite Volume Method described in the class.
        """
        raise NotImplementedError

    @abstractmethod
    def limited_solution_at_quadrature_point(
        self,
        refBLK: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def unlimited_solution_at_quadrature_point(
        self,
        refBLK: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> np.ndarray:
        raise NotImplementedError

    def dUdt(self, refBLK: QuadBlock):
        """
        Compute residuals used for marching the solution through time by integrating the fluxes on each cell face and
        applying the semi-discrete Godunov method:

        dUdt[i] = - (1/A[i]) * sum[over all faces] (F[face] * length[face])
        """
        refBLK.reconBlk.from_block(refBLK)
        refBLK.reconBlk.fvm.evaluate_flux(refBLK.reconBlk)
        fluxE = self.integrate_flux(
            face_length=refBLK.mesh.face.E.L,
            quadrature_points=refBLK.reconBlk.qp.E,
            fluxes=refBLK.reconBlk.fvm.Flux.E,
        )
        fluxW = self.integrate_flux(
            face_length=refBLK.mesh.face.W.L,
            quadrature_points=refBLK.reconBlk.qp.W,
            fluxes=refBLK.reconBlk.fvm.Flux.W,
        )
        fluxN = self.integrate_flux(
            face_length=refBLK.mesh.face.N.L,
            quadrature_points=refBLK.reconBlk.qp.N,
            fluxes=refBLK.reconBlk.fvm.Flux.N,
        )
        fluxS = self.integrate_flux(
            face_length=refBLK.mesh.face.S.L,
            quadrature_points=refBLK.reconBlk.qp.S,
            fluxes=refBLK.reconBlk.fvm.Flux.S,
        )
        if self.config.use_JIT:
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

    @staticmethod
    def integrate_flux(
        face_length: np.ndarray,
        quadrature_points: tuple[QuadraturePoint],
        fluxes: tuple[np.ndarray],
    ) -> np.ndarray:
        """
        Integrates the fluxes using an n-point gauss gradrature rule.

        :type face_length: np.ndarray
        :param face_length: Reference block with flux data for integration

        :type quadrature_points: tuple[QuadraturePoint]
        :param quadrature_points: tuple of quadrature point objects

        :type fluxes: tuple[np.ndarray]
        :param fluxes: tuple of fluxes for each quadrature point

        :rtype: np.ndarray
        :return: South face integrated fluxes
        """
        return (
            0.5
            * face_length
            * sum((qp.w * qpflux for (qp, qpflux) in zip(quadrature_points, fluxes)))
        )


class MUSCL(FiniteVolumeMethod, ABC):
    def __init__(
        self,
        config,
        flux: [FluxFunction],
        limiter: SlopeLimiter,
        gradient: Gradient,
    ) -> None:
        """
        Monotonic Upstream-centered Scheme for Conservation Laws.

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
        super().__init__(config=config, flux=flux)

        # Flux storage arrays
        self.Flux = SidePropertyContainer(
            E=list(
                np.empty((self.config.ny, self.config.nx, 4))
                for _ in range(self.config.fvm_num_quadrature_points)
            ),
            W=list(
                np.empty((self.config.ny, self.config.nx, 4))
                for _ in range(self.config.fvm_num_quadrature_points)
            ),
            N=list(
                np.empty((self.config.ny, self.config.nx, 4))
                for _ in range(self.config.fvm_num_quadrature_points)
            ),
            S=list(
                np.empty((self.config.ny, self.config.nx, 4))
                for _ in range(self.config.fvm_num_quadrature_points)
            ),
        )
        self.limiter = limiter
        self.gradient = gradient

    def _get_LR_states_for_flux_calc(
        self,
        ghostL: State,
        stateL: State,
        ghostR: State,
        stateR: State,
    ) -> [PrimitiveState]:
        """
        Compute and return the left and right states used for the North-South inviscid flux calculation. The left state
        is created by concatenating the south ghost-cell state and the north face state. The right state
        is created by concatenating the north ghost-cell state and the south face state. After concatenation, the arrays
        are reshaped to produce a (1, n, 4) shaped array.

        :type ghostL: np.ndarray
        :param ghostL: left ghost cell state array

        :type stateL: np.ndarray
        :param stateL: left state array

        :type ghostR: np.ndarray
        :param ghostR: right ghost cell state array

        :type stateR: np.ndarray
        :param stateR: right state array

        :rtype: tuple(PrimitiveState, PrimitiveState)
        :return: PrimitiveStates that hold the left and right states for the flux calculation
        """
        left_arr = np.concatenate((ghostL.data, stateL.data), axis=1)
        right_arr = np.concatenate((stateR.data, ghostR.data), axis=1)

        if self.config.reconstruction_type is PrimitiveState:
            left_state = PrimitiveState(self.config.fluid, array=left_arr)
            right_state = PrimitiveState(self.config.fluid, array=right_arr)
            return left_state, right_state

        left_state = PrimitiveState(
            fluid=self.config.fluid,
            state=self.config.reconstruction_type(
                fluid=self.config.fluid, array=left_arr
            ),
        )
        right_state = PrimitiveState(
            fluid=self.config.fluid,
            state=self.config.reconstruction_type(
                fluid=self.config.fluid, array=right_arr
            ),
        )
        return left_state, right_state

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
            (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK,
                    slicer=self.east_boundary_slice,
                )
                for qe in refBLK.qp.E
            )
            if refBLK.ghost.E.BCtype is not None
            else (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.E,
                    slicer=self.west_boundary_slice,
                )
                for qe in refBLK.ghost.E.qp.E
            )
        )

        bndW = (
            (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK,
                    slicer=self.west_boundary_slice,
                )
                for qe in refBLK.qp.E
            )
            if refBLK.ghost.W.BCtype is not None
            else (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.W,
                    slicer=self.east_boundary_slice,
                )
                for qe in refBLK.ghost.W.qp.W
            )
        )

        for qe, qw, _bndE, _bndW, fluxE, fluxW in zip(
            refBLK.qp.E, refBLK.qp.W, bndE, bndW, self.Flux.E, self.Flux.W
        ):
            _stateE = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK, qp=qe,
            )
            _stateW = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK, qp=qw,
            )
            refBLK.ghost.E.apply_boundary_condition(_bndE)
            refBLK.ghost.W.apply_boundary_condition(_bndW)

            if not refBLK.is_cartesian:
                utils.rotate(refBLK.mesh.face.E.theta, _stateE.data)
                utils.rotate(refBLK.mesh.face.W.theta, _stateW.data)
                utils.rotate(refBLK.mesh.east_boundary_angle(), _bndE.data)
                utils.rotate(refBLK.mesh.west_boundary_angle(), _bndW.data)

            sL, sR = self._get_LR_states_for_flux_calc(
                ghostL=_bndW, stateL=_stateE, ghostR=_bndE, stateR=_stateW
            )
            fluxEW = self.flux_function_x(WL=sL, WR=sR)
            fluxE[:] = fluxEW[:, 1:, :]
            fluxW[:] = fluxEW[:, :-1, :]

            if not refBLK.is_cartesian:
                utils.unrotate(refBLK.mesh.face.E.theta, fluxE)
                utils.unrotate(refBLK.mesh.face.W.theta, fluxW)

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
            (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK,
                    slicer=self.north_boundary_slice,
                )
                for qe in refBLK.qp.N
            )
            if refBLK.ghost.N.BCtype is not None
            else (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.N,
                    slicer=self.south_boundary_slice,
                )
                for qe in refBLK.ghost.N.qp.N
            )
        )

        bndS = (
            (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK,
                    slicer=self.south_boundary_slice,
                )
                for qe in refBLK.qp.S
            )
            if refBLK.ghost.S.BCtype is not None
            else (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.S,
                    slicer=self.north_boundary_slice,
                )
                for qe in refBLK.ghost.S.qp.S
            )
        )

        for qn, qs, _bndN, _bndS, fluxN, fluxS in zip(
            refBLK.qp.N, refBLK.qp.S, bndN, bndS, self.Flux.N, self.Flux.S
        ):
            _stateN = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK, qp=qn,
            )
            _stateS = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK, qp=qs,
            )
            refBLK.ghost.N.apply_boundary_condition(_bndN)
            refBLK.ghost.S.apply_boundary_condition(_bndS)

            if refBLK.is_cartesian:
                utils.rotate90(_stateN.data, _stateS.data, _bndN.data, _bndS.data)
            else:
                utils.rotate(refBLK.mesh.face.N.theta, _stateN.data)
                utils.rotate(refBLK.mesh.face.S.theta, _stateS.data)
                utils.rotate(refBLK.mesh.north_boundary_angle(), _bndN.data)
                utils.rotate(refBLK.mesh.south_boundary_angle(), _bndS.data)

            # Transpose to x-frame
            _bndS.transpose((1, 0, 2))
            _stateN.transpose((1, 0, 2))
            _bndN.transpose((1, 0, 2))
            _stateS.transpose((1, 0, 2))

            sL, sR = self._get_LR_states_for_flux_calc(
                _bndS,
                _stateN,
                _bndN,
                _stateS,
            )
            fluxNS = self.flux_function_y(WL=sL, WR=sR).transpose((1, 0, 2))
            fluxN[:] = fluxNS[1:, :, :]
            fluxS[:] = fluxNS[:-1, :, :]

            if refBLK.is_cartesian:
                utils.unrotate90(fluxN, fluxS)
            else:
                utils.unrotate(refBLK.mesh.face.N.theta, fluxN)
                utils.unrotate(refBLK.mesh.face.S.theta, fluxS)

    def evaluate_flux(self, refBLK: QuadBlock) -> None:
        """
        Calculates the fluxes at all cell boundaries. Solves the 1-D riemann problem along all of the rows and columns
        of cells on the blocks in a sweeping (but unsplit) fashion.

        :type refBLK: QuadBlock
        :param refBLK: QuadBlock that holds the solution data for the flux calculation

        :rtype: None
        :return: None
        """

        self.gradient.compute(refBLK)
        self.compute_limiter(refBLK)
        self.evaluate_flux_EW(refBLK)
        self.evaluate_flux_NS(refBLK)
