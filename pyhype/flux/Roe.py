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
import os

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np

np.set_printoptions(linewidth=200)
np.set_printoptions(precision=3)
from scipy.sparse import coo_matrix as coo
from pyhype.flux.base import FluxFunction
from pyhype.states.primitive import RoePrimitiveState, PrimitiveState
from pyhype.flux.eigen_system import XDIR_EIGENSYSTEM_INDICES, XDIR_EIGENSYSTEM_VECTORS


class FluxRoe(FluxFunction):
    def __init__(self, config, size, sweeps):
        super().__init__(config, nx=size, ny=sweeps)
        # Thermodynamic quantities
        g = self.config.fluid.gamma()
        self.gh = self.config.fluid.gamma() - 1
        self.gt = 1 - g
        self.gb = 3 - g
        # Matrix size
        self.size = size
        self.sweeps = sweeps
        self.num = (self.size + 1) * self.sweeps

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self.config, size, sweeps)
        self.A_d0, self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 = (
            vec.A_d0,
            vec.A_m1,
            vec.A_m2,
            vec.A_m3,
            vec.A_p1,
            vec.A_p2,
        )
        (
            self.Rc_d0,
            self.Rc_m1,
            self.Rc_m2,
            self.Rc_m3,
            self.Rc_p1,
            self.Rc_p2,
            self.Rc_p3,
        ) = (
            vec.Rc_d0,
            vec.Rc_m1,
            vec.Rc_m2,
            vec.Rc_m3,
            vec.Rc_p1,
            vec.Rc_p2,
            vec.Rc_p3,
        )
        self.Lp_m2, self.Lp_m1, self.Lp_d0, self.Lp_p1, self.Lp_p2, self.Lp_p3 = (
            vec.Lp_m2,
            vec.Lp_m1,
            vec.Lp_d0,
            vec.Lp_p1,
            vec.Lp_p2,
            vec.Lp_p3,
        )
        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(size, sweeps)
        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Rci, self.Rcj = idx.Rci, idx.Rcj
        self.Lpi, self.Lpj = idx.Lpi, idx.Lpj
        self.Li, self.Lj = idx.Li, idx.Lj

        # Flux jacobian data container
        A_data = np.concatenate(
            (
                self.A_m3,  # Subdiagonals
                self.A_m2,  # Subdiagonals
                self.A_m1,  # Subdiagonals
                self.A_d0,  # Diagonal
                self.A_p1,  # Superdiagonals
                self.A_p2,  # Superdiagonals
            )
        )
        # Right eigenvectors data container
        Rc_data = np.concatenate(
            (
                self.Rc_m3,  # Subdiagonals
                self.Rc_m2,  # Subdiagonals
                self.Rc_m1,  # Subdiagonals
                self.Rc_d0,  # Diagonal
                self.Rc_p1,  # Superdiagonals
                self.Rc_p2,  # Superdiagonals
                self.Rc_p3,  # Superdiagonals
            )
        )
        # Left eigenvectors primitive data container
        Lp_data = np.concatenate(
            (
                self.Lp_m2,  # Subdiagonals
                self.Lp_m1,  # Subdiagonals
                self.Lp_d0,  # Diagonal
                self.Lp_p1,  # Superdiagonals
                self.Lp_p2,  # Superdiagonals
                self.Lp_p3,  # Superdiagonals
            )
        )
        # Eigenvalue data container
        L_data = np.zeros((4 * self.num), dtype=float)

        # Build sparse matrices
        self.Jac = coo((A_data, (self.Ai, self.Aj)))
        self.Rc = coo((Rc_data, (self.Rci, self.Rcj)))
        self.Lp = coo((Lp_data, (self.Lpi, self.Lpj)))
        self.Lambda = coo((L_data, (self.Li, self.Lj)))

    def compute_flux_jacobian(
        self,
        Wroe: RoePrimitiveState,
        uv: np.ndarray = None,
        ghv: np.ndarray = None,
    ) -> None:
        """
        Calculate the x-direction sparse flux jacobian matrix A. A is defined as:

        Parameters:
            - Wroe (RoePrimitiveState): Roe primitive state object, contains the roe average state calculated using
            - Ek (np.ndarray): (Optional) Kinetic energy = 0.5 * (v^2 + u^2)
            - H (np.ndarray): (Optional) Total enthalpy
            - uv (np.ndarray): (Optional) u velocity * v velocity
            - u2 (np.ndarray): (Optional) u velocity squared
            - ghv (np.ndarray): (Optional) gamma_hat * v velocity

        Returns:
            - N.A
        """

        # Compute optional parameters if they are not passed.
        ghv = self.gh * Wroe.v if ghv is None else ghv
        uv = Wroe.u * Wroe.v if uv is None else uv
        Ek = Wroe.Ek()
        H = Wroe.H()
        u2 = Wroe.u**2
        ghek = self.gh * Ek

        # -3 subdiagonal entries
        self.Jac.data[0 * self.num : 1 * self.num] = Wroe.u * (ghek - H)
        # -2 subdiagonal entries
        self.Jac.data[1 * self.num : 2 * self.num] = -uv
        self.Jac.data[2 * self.num : 3 * self.num] = H - self.gh * u2
        # -1 subdiagonal entries
        self.Jac.data[3 * self.num : 4 * self.num] = ghek - u2
        self.Jac.data[4 * self.num : 5 * self.num] = Wroe.v
        self.Jac.data[5 * self.num : 6 * self.num] = -self.gh * uv
        # diagonal entries
        self.Jac.data[6 * self.num : 7 * self.num] = self.gb * Wroe.u
        self.Jac.data[7 * self.num : 8 * self.num] = Wroe.u
        self.Jac.data[8 * self.num : 9 * self.num] = self.g * Wroe.u
        # +1 subdiagonal entries
        self.Jac.data[10 * self.num : 11 * self.num] = -ghv

    def compute_Rc(
        self,
        Wroe: RoePrimitiveState,
        Lm: np.ndarray,
        Lp: np.ndarray,
        ua: np.ndarray = None,
    ) -> None:
        """
        Computes the right eigenvectors of the euler equations linear eigensystem.

        :type Wroe: PrimitiveState
        :param Wroe: Roe primitive state

        :type Lm: np.ndarray
        :param Lm: Slow wave speed

        :type Lp: np.ndarray
        :param Lp: Fast wave speed

        :type ua: np.ndarray
        :param ua: flow velocity * speed of sound

        :rtype: None
        :return: None
        """

        Ek = Wroe.Ek()
        H = Wroe.H(Ek=Ek)
        ua = Wroe.u * Wroe.a() if ua is None else ua

        # -3 subdiagonal entries
        self.Rc.data[0 * self.num : 1 * self.num] = H - ua
        # -2 subdiagonal entries
        self.Rc.data[1 * self.num : 2 * self.num] = Wroe.v
        self.Rc.data[2 * self.num : 3 * self.num] = Ek
        # -1 subdiagonal entries
        self.Rc.data[3 * self.num : 4 * self.num] = Lm
        self.Rc.data[4 * self.num : 5 * self.num] = Wroe.v
        self.Rc.data[5 * self.num : 6 * self.num] = Wroe.v
        # diagonal entries
        self.Rc.data[7 * self.num : 8 * self.num] = Wroe.u
        self.Rc.data[9 * self.num : 10 * self.num] = H + ua
        # +1 subdiagonal entries
        self.Rc.data[11 * self.num : 12 * self.num] = Wroe.v
        # +2 subdiagonal entries
        self.Rc.data[12 * self.num : 13 * self.num] = Lp

    def compute_Lp(
        self,
        Wroe: RoePrimitiveState,
    ) -> None:

        """
        Computes the left eigenvectors of the euler equations linear eigensystem.

        :type Wroe: PrimitiveState
        :param Wroe: Roe primitive state

        :rtype: None
        :return: None
        """

        _a = 1.0 / Wroe.a()
        one_a2 = _a * _a
        h_one_a2 = 0.5 * one_a2
        r2a = 0.5 * Wroe.rho * _a

        # -2 subdiagonal entries
        self.Lp.data[0 * self.num : 1 * self.num] = r2a
        # -1 subdiagonal entries
        self.Lp.data[3 * self.num : 4 * self.num] = h_one_a2
        # +1 subdiagonal entries
        self.Lp.data[4 * self.num : 5 * self.num] = -r2a
        # +2 subdiagonal entries
        self.Lp.data[5 * self.num : 6 * self.num] = -one_a2
        # +3 subdiagonal entries
        self.Lp.data[6 * self.num : 7 * self.num] = h_one_a2

    def compute_eigenvalues(
        self, Wroe: RoePrimitiveState, Lp: np.ndarray, Lm: np.ndarray
    ) -> None:

        self.Lambda.data[0 * self.num : 1 * self.num] = Lm
        self.Lambda.data[1 * self.num : 2 * self.num] = Wroe.u
        self.Lambda.data[2 * self.num : 3 * self.num] = Wroe.u
        self.Lambda.data[3 * self.num : 4 * self.num] = Lp

    def compute_flux(self, WL: PrimitiveState, WR: PrimitiveState) -> np.ndarray:
        """
        Computes the flux using the Roe approximate riemann solver. First, the Roe average state is computed based on
        the given left and right states, and then, the euler eigensystem is computed by diagonalization. The
        eigensystem is then used to compute the flux via the Roe flux function.
        """
        new_shape = (1, (self.nx + 1) * self.ny, 4)
        WR.reshape(new_shape)
        WL.reshape(new_shape)
        flux = 0.5 * (WL.F() + WR.F()) - self.get_upwind_flux(WL, WR)
        return flux.reshape((self.ny, (self.nx + 1), 4))

    def get_upwind_flux(
        self,
        WL: PrimitiveState,
        WR: PrimitiveState,
    ) -> np.ndarray:
        Wroe = RoePrimitiveState(self.config.fluid, WL, WR)
        self.diagonalize(Wroe, WL, WR)
        dW = (WR.data - WL.data).flatten()
        self.Lambda.data = np.absolute(self.Lambda.data)
        return 0.5 * (self.Rc * (self.Lambda * (self.Lp * dW))).reshape(1, -1, 4)

    def diagonalize(
        self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState
    ) -> None:
        """
        Diagonalization of the Euler equations in primitive form. First, the Harten correction is applied to prevent
        expansion shocks, and then, the right-conservative, left-primitive primitive eigenvectors, and eigenvalues are
        calculated.

        Parameters:
            - Wroe (RoePrimitiveState): Primitive state holding the Roe averages
            - WL (PrimitiveState): Primitive left states
            - WR (PrimitiveState): Primitive right states

        Returns:
            - None
        """
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)
        self.compute_Rc(Wroe, Lm, Lp)
        self.compute_Lp(Wroe)
        self.compute_eigenvalues(Wroe, Lp, Lm)
