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

    def dUdt(self, refBLK: QuadBlock) -> np.ndarray:
        """
        Compute residuals used for marching the solution through time by integrating the fluxes on each cell face and
        applying the semi-discrete Godunov method:

        dUdt[i] = - (1/A[i]) * sum[over all faces] (F[face] * length[face])
        """
        refBLK.reconBlk.from_block(refBLK)
        refBLK.reconBlk.fvm.evaluate_flux(refBLK.reconBlk)
        integrated_east_flux = self.integrate_flux(
            fluxes=refBLK.reconBlk.fvm.Flux.E,
            face_length=refBLK.mesh.face.E.L,
            quadrature_points=refBLK.reconBlk.qp.E,
        )
        integrated_west_flux = self.integrate_flux(
            fluxes=refBLK.reconBlk.fvm.Flux.W,
            face_length=refBLK.mesh.face.W.L,
            quadrature_points=refBLK.reconBlk.qp.W,
        )
        integrated_north_flux = self.integrate_flux(
            fluxes=refBLK.reconBlk.fvm.Flux.N,
            face_length=refBLK.mesh.face.N.L,
            quadrature_points=refBLK.reconBlk.qp.N,
        )
        integrated_south_flux = self.integrate_flux(
            fluxes=refBLK.reconBlk.fvm.Flux.S,
            face_length=refBLK.mesh.face.S.L,
            quadrature_points=refBLK.reconBlk.qp.S,
        )
        if self.config.use_JIT:
            return self._dUdt_JIT(
                integrated_east_flux,
                integrated_west_flux,
                integrated_north_flux,
                integrated_south_flux,
                refBLK.mesh.A,
            )
        return (
            integrated_west_flux
            - integrated_east_flux
            + integrated_south_flux
            - integrated_north_flux
        ) / refBLK.mesh.A

    @staticmethod
    @nb.njit(cache=True)
    def _dUdt_JIT(
        east_flux: np.ndarray,
        west_flux: np.ndarray,
        north_flux: np.ndarray,
        south_flux: np.ndarray,
        A: np.ndarray,
    ) -> np.ndarray:
        dUdt = np.zeros_like(east_flux)
        for i in range(east_flux.shape[0]):
            for j in range(east_flux.shape[1]):
                a = A[i, j, 0]
                for k in range(east_flux.shape[2]):
                    dUdt[i, j, k] = (
                        west_flux[i, j, k]
                        - east_flux[i, j, k]
                        + south_flux[i, j, k]
                        - north_flux[i, j, k]
                    ) / a
        return dUdt

    @staticmethod
    def integrate_flux(
        fluxes: tuple[np.ndarray],
        face_length: np.ndarray,
        quadrature_points: tuple[QuadraturePoint],
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

    def _get_left_right_riemann_states(
        self,
        right_state: State,
        left_state: State,
        right_ghost_state: State,
        left_ghost_state: State,
    ) -> [PrimitiveState]:
        """
        Compute and return the left and right states used for the North-South inviscid flux calculation. The left state
        is created by concatenating the south ghost-cell state and the north face state. The right state
        is created by concatenating the north ghost-cell state and the south face state. After concatenation, the arrays
        are reshaped to produce a (1, n, 4) shaped array.

        :type left_ghost_state: np.ndarray
        :param left_ghost_state: left ghost cell state array

        :type left_state: np.ndarray
        :param left_state: left state array

        :type right_ghost_state: np.ndarray
        :param right_ghost_state: right ghost cell state array

        :type right_state: np.ndarray
        :param right_state: right state array

        :rtype: tuple(PrimitiveState, PrimitiveState)
        :return: PrimitiveStates that hold the left and right states for the flux calculation
        """
        left_arr = np.concatenate((left_ghost_state.data, left_state.data), axis=1)
        right_arr = np.concatenate((right_state.data, right_ghost_state.data), axis=1)

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

    def _get_east_flux_states(self, refBLK: QuadBlock):
        if refBLK.ghost.E.BCtype is None:
            return (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.E,
                    slicer=self.west_boundary_slice,
                )
                for qe in refBLK.qp.E
            )
        return (
            self.limited_solution_at_quadrature_point(
                qp=qe,
                refBLK=refBLK,
                slicer=self.east_boundary_slice,
            )
            for qe in refBLK.qp.E
        )

    def _get_west_flux_states(self, refBLK: QuadBlock):
        if refBLK.ghost.W.BCtype is None:
            return (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.W,
                    slicer=self.east_boundary_slice,
                )
                for qe in refBLK.qp.W
            )
        return (
            self.limited_solution_at_quadrature_point(
                qp=qe,
                refBLK=refBLK,
                slicer=self.west_boundary_slice,
            )
            for qe in refBLK.qp.W
        )

    def _evaluate_east_west_flux(self, refBLK: QuadBlock) -> None:
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
        east_boundary_states = self._get_east_flux_states(refBLK=refBLK)
        west_boundary_states = self._get_west_flux_states(refBLK=refBLK)

        for qe, qw, east_boundary, west_boundary, east_flux, west_flux in zip(
            refBLK.qp.E,
            refBLK.qp.W,
            east_boundary_states,
            west_boundary_states,
            self.Flux.E,
            self.Flux.W,
        ):
            east_face_states = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK,
                qp=qe,
            )
            west_face_states = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK,
                qp=qw,
            )

            if refBLK.ghost.E.BCtype is not None:
                refBLK.ghost.E.apply_boundary_condition(east_boundary)
            if refBLK.ghost.W.BCtype is not None:
                refBLK.ghost.W.apply_boundary_condition(west_boundary)

            if not refBLK.is_cartesian:
                utils.rotate(refBLK.mesh.face.E.theta, east_face_states.data)
                utils.rotate(refBLK.mesh.face.W.theta, west_face_states.data)
                utils.rotate(refBLK.mesh.east_boundary_angle(), east_boundary.data)
                utils.rotate(refBLK.mesh.west_boundary_angle(), west_boundary.data)

            left, right = self._get_left_right_riemann_states(
                right_state=west_face_states,
                left_state=east_face_states,
                right_ghost_state=east_boundary,
                left_ghost_state=west_boundary,
            )
            east_west_flux = self.flux_function_x(WL=left, WR=right)
            east_flux[:] = east_west_flux[:, 1:, :]
            west_flux[:] = east_west_flux[:, :-1, :]

            if not refBLK.is_cartesian:
                utils.unrotate(refBLK.mesh.face.E.theta, east_flux)
                utils.unrotate(refBLK.mesh.face.W.theta, west_flux)

    def _get_north_flux_states(self, refBLK: QuadBlock):
        if refBLK.ghost.N.BCtype is None:
            return (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.N,
                    slicer=self.south_boundary_slice,
                )
                for qe in refBLK.qp.N
            )
        return (
            self.limited_solution_at_quadrature_point(
                qp=qe,
                refBLK=refBLK,
                slicer=self.north_boundary_slice,
            )
            for qe in refBLK.qp.N
        )

    def _get_south_flux_states(self, refBLK: QuadBlock):
        if refBLK.ghost.S.BCtype is None:
            return (
                self.limited_solution_at_quadrature_point(
                    qp=qe,
                    refBLK=refBLK.ghost.S,
                    slicer=self.north_boundary_slice,
                )
                for qe in refBLK.qp.S
            )
        return (
            self.limited_solution_at_quadrature_point(
                qp=qe,
                refBLK=refBLK,
                slicer=self.south_boundary_slice,
            )
            for qe in refBLK.qp.S
        )

    def _evaluate_north_south_flux(self, refBLK: QuadBlock) -> None:
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
        north_boundary_states = self._get_north_flux_states(refBLK=refBLK)
        south_boundary_states = self._get_south_flux_states(refBLK=refBLK)

        for qn, qs, north_boundary, south_boundary, north_flux, south_flux in zip(
            refBLK.qp.N,
            refBLK.qp.S,
            north_boundary_states,
            south_boundary_states,
            self.Flux.N,
            self.Flux.S,
        ):
            north_face_states = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK,
                qp=qn,
            )
            south_face_states = refBLK.fvm.limited_solution_at_quadrature_point(
                refBLK=refBLK,
                qp=qs,
            )

            if refBLK.ghost.N.BCtype is not None:
                refBLK.ghost.N.apply_boundary_condition(north_boundary)
            if refBLK.ghost.S.BCtype is not None:
                refBLK.ghost.S.apply_boundary_condition(south_boundary)

            if refBLK.is_cartesian:
                utils.rotate90(
                    north_face_states.data,
                    south_face_states.data,
                    north_boundary.data,
                    south_boundary.data,
                )
            else:
                utils.rotate(refBLK.mesh.face.N.theta, north_face_states.data)
                utils.rotate(refBLK.mesh.face.S.theta, south_face_states.data)
                utils.rotate(refBLK.mesh.north_boundary_angle(), north_boundary.data)
                utils.rotate(refBLK.mesh.south_boundary_angle(), south_boundary.data)

            # Transpose to x-frame
            south_boundary.transpose((1, 0, 2))
            north_face_states.transpose((1, 0, 2))
            north_boundary.transpose((1, 0, 2))
            south_face_states.transpose((1, 0, 2))

            left, right = self._get_left_right_riemann_states(
                right_state=south_face_states,
                left_state=north_face_states,
                right_ghost_state=north_boundary,
                left_ghost_state=south_boundary,
            )
            north_south_flux = self.flux_function_y(WL=left, WR=right).transpose(
                (1, 0, 2)
            )
            north_flux[:] = north_south_flux[1:, :, :]
            south_flux[:] = north_south_flux[:-1, :, :]

            if refBLK.is_cartesian:
                utils.unrotate90(north_flux, south_flux)
            else:
                utils.unrotate(refBLK.mesh.face.N.theta, north_flux)
                utils.unrotate(refBLK.mesh.face.S.theta, south_flux)

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
        self._evaluate_east_west_flux(refBLK)
        self._evaluate_north_south_flux(refBLK)
