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
from pyhype.utils.utils import SidePropertyDict
from pyhype.states.primitive import PrimitiveState
from pyhype.utils.utils import Direction

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
        parent_block: QuadBlock,
    ):
        self.config = config
        self.flux_function_x, self.flux_function_y = flux
        self.parent_block = parent_block

    @abstractmethod
    def compute_limiter(self) -> [np.ndarray]:
        """
        Implementation of the reconstruction method specialized to the Finite Volume Method described in the class.
        """
        raise NotImplementedError

    @abstractmethod
    def limited_solution_at_quadrature_point(
        self,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> State:
        raise NotImplementedError

    @abstractmethod
    def unlimited_solution_at_quadrature_point(
        self,
        parent_block: BaseBlockFVM,
        qp: QuadraturePoint,
        slicer: Union[slice, tuple, int] = NumpySlice.all(),
    ) -> State:
        raise NotImplementedError

    def dUdt(self) -> np.ndarray:
        """
        Compute residuals used for marching the solution through time by integrating the fluxes on each cell face and
        applying the semi-discrete Godunov method:

        dUdt[i] = - (1/A[i]) * sum[over all faces] (F[face] * length[face])
        """
        self.parent_block.recon_block.from_block(self.parent_block)
        self.parent_block.recon_block.fvm.evaluate_flux()

        integrated_fluxes = SidePropertyDict(
            *(
                self.integrate_flux(
                    fluxes=flux,
                    face_length=face.L,
                    quadrature_points=qp,
                )
                for flux, face, qp in zip(
                    self.parent_block.recon_block.fvm.Flux.values(),
                    self.parent_block.recon_block.mesh.face.values(),
                    self.parent_block.recon_block.qp.values(),
                )
            )
        )

        return self._dUdt_JIT(
            integrated_fluxes.E,
            integrated_fluxes.W,
            integrated_fluxes.N,
            integrated_fluxes.S,
            self.parent_block.recon_block.mesh.A,
        )

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
                        0.5
                        * (
                            west_flux[i, j, k]
                            - east_flux[i, j, k]
                            + south_flux[i, j, k]
                            - north_flux[i, j, k]
                        )
                        / a
                    )
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
        return face_length * sum(
            (qp.w * qpflux for (qp, qpflux) in zip(quadrature_points, fluxes))
        )


class MUSCL(FiniteVolumeMethod, ABC):
    def __init__(
        self,
        config,
        flux: [FluxFunction],
        limiter: SlopeLimiter,
        gradient: Gradient,
        parent_block: QuadBlock,
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
        super().__init__(config=config, flux=flux, parent_block=parent_block)

        # Flux storage arrays
        self.Flux = SidePropertyDict(
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
        self._boundary_index = NumpySlice.boundary()

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

    def _get_boundary_flux_states(self, direction: int) -> [State]:
        """
        Get a generator of States that contain the limited solution
        at the east block boundary wuadrature points.

        :rtype: [State]
        :return: Gen exp of State objects
        """
        slicer = self._boundary_index[direction]
        func = self.limited_solution_at_quadrature_point
        if self.parent_block.ghost[direction].bc_type is None:
            slicer = self._boundary_index[-direction]
            func = self.parent_block.ghost[
                direction
            ].fvm.limited_solution_at_quadrature_point
        return (func(qp=qe, slicer=slicer) for qe in self.parent_block.qp[direction])

    def _evaluate_east_west_flux(self) -> None:
        """
        Evaluates the fluxes at each east-west cell boundary. The following steps are followed:
            1. Get list of reconstructed boundary states at each quadrature point on the east boundary
            2. Get list of reconstructed boundary states at each quadrature point on the west boundary
            3. Loop through each quadrature point on the east and west faces and calculate the reconstructed solution
               states at each q-point
            4. Rotate states if the block is not cartesian
            5. Compute left and right states
            6. Compute fluxes

        :rtype: None
        :return: None
        """
        east_boundary_states = self._get_boundary_flux_states(direction=Direction.east)
        west_boundary_states = self._get_boundary_flux_states(direction=Direction.west)

        for qe, qw, east_boundary, west_boundary, east_flux, west_flux in zip(
            self.parent_block.qp.E,
            self.parent_block.qp.W,
            east_boundary_states,
            west_boundary_states,
            self.Flux.E,
            self.Flux.W,
        ):
            east_face_states = self.limited_solution_at_quadrature_point(qp=qe)
            west_face_states = self.limited_solution_at_quadrature_point(qp=qw)

            if self.parent_block.ghost.E.bc_type is not None:
                self.parent_block.ghost.E.apply_boundary_condition_to_state(
                    east_boundary
                )
            if self.parent_block.ghost.W.bc_type is not None:
                self.parent_block.ghost.W.apply_boundary_condition_to_state(
                    west_boundary
                )

            if not self.parent_block.is_cartesian:
                utils.rotate(self.parent_block.mesh.face.E.theta, east_face_states.data)
                utils.rotate(self.parent_block.mesh.face.W.theta, west_face_states.data)
                utils.rotate(
                    self.parent_block.mesh.boundary_angle(direction=Direction.east),
                    east_boundary.data,
                )
                utils.rotate(
                    self.parent_block.mesh.boundary_angle(direction=Direction.west),
                    west_boundary.data,
                )

            left, right = self._get_left_right_riemann_states(
                right_state=west_face_states,
                left_state=east_face_states,
                right_ghost_state=east_boundary,
                left_ghost_state=west_boundary,
            )
            east_west_flux = self.flux_function_x(WL=left, WR=right)
            east_flux[:] = east_west_flux[:, 1:, :]
            west_flux[:] = east_west_flux[:, :-1, :]

            if not self.parent_block.is_cartesian:
                utils.unrotate(self.parent_block.mesh.face.E.theta, east_flux)
                utils.unrotate(self.parent_block.mesh.face.W.theta, west_flux)

    def _evaluate_north_south_flux(self) -> None:
        """
        Evaluates the fluxes at each north-south cell boundary. The following steps are followed:
            1. Get list of reconstructed boundary states at each quadrature point on the north boundary
            2. Get list of reconstructed boundary states at each quadrature point on the south boundary
            3. Loop through each quadrature point on the north and south faces and calculate the reconstructed solution
               states at each q-point
            4. Rotate states if the block is not cartesian
            5. Compute left and right states
            6. Compute fluxes

        :rtype: None
        :return: None
        """
        north_boundary_states = self._get_boundary_flux_states(
            direction=Direction.north
        )
        south_boundary_states = self._get_boundary_flux_states(
            direction=Direction.south
        )

        for qn, qs, north_boundary, south_boundary, north_flux, south_flux in zip(
            self.parent_block.qp.N,
            self.parent_block.qp.S,
            north_boundary_states,
            south_boundary_states,
            self.Flux.N,
            self.Flux.S,
        ):
            north_face_states = self.limited_solution_at_quadrature_point(qp=qn)
            south_face_states = self.limited_solution_at_quadrature_point(qp=qs)

            if self.parent_block.ghost.N.bc_type is not None:
                self.parent_block.ghost.N.apply_boundary_condition_to_state(
                    north_boundary
                )
            if self.parent_block.ghost.S.bc_type is not None:
                self.parent_block.ghost.S.apply_boundary_condition_to_state(
                    south_boundary
                )

            if self.parent_block.is_cartesian:
                utils.rotate90(
                    north_face_states.data,
                    south_face_states.data,
                    north_boundary.data,
                    south_boundary.data,
                )
            else:
                utils.rotate(
                    self.parent_block.mesh.face.N.theta, north_face_states.data
                )
                utils.rotate(
                    self.parent_block.mesh.face.S.theta, south_face_states.data
                )
                utils.rotate(
                    self.parent_block.mesh.boundary_angle(direction=Direction.north),
                    north_boundary.data,
                )
                utils.rotate(
                    self.parent_block.mesh.boundary_angle(direction=Direction.south),
                    south_boundary.data,
                )

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

            if self.parent_block.is_cartesian:
                utils.unrotate90(north_flux, south_flux)
            else:
                utils.unrotate(self.parent_block.mesh.face.N.theta, north_flux)
                utils.unrotate(self.parent_block.mesh.face.S.theta, south_flux)

    def evaluate_flux(self) -> None:
        """
        Calculates the fluxes at all cell boundaries. Solves the 1-D riemann problem along all of the rows and columns
        of cells on the blocks in a sweeping (but unsplit) fashion.

        :rtype: None
        :return: None
        """

        self.gradient.compute(self.parent_block)
        self.compute_limiter()
        self._evaluate_east_west_flux()
        self._evaluate_north_south_flux()
