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

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=3)
from scipy.sparse import coo_matrix as coo
from pyHype.flux.base import FluxFunction
from pyHype.states.states import RoePrimitiveState, ConservativeState, PrimitiveState
from pyHype.flux.eigen_system import XDIR_EIGENSYSTEM_INDICES, XDIR_EIGENSYSTEM_VECTORS


class FluxRoe(FluxFunction):
    def __init__(self, inputs, size, sweeps):
        super().__init__(inputs)
        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g
        # Matrix size
        self.size = size
        self.sweeps = sweeps

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self.inputs, size, sweeps)
        self.A_d0, self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 \
            = vec.A_d0, vec.A_m1, vec.A_m2, vec.A_m3, vec.A_p1, vec.A_p2
        self.Rc_d0, self.Rc_m1, self.Rc_m2, self.Rc_m3, self.Rc_p1, self.Rc_p2, self.Rc_p3 \
            = vec.Rc_d0, vec.Rc_m1, vec.Rc_m2, vec.Rc_m3, vec.Rc_p1, vec.Rc_p2, vec.Rc_p3
        self.Lc_d0, self.Lc_m1, self.Lc_m2, self.Lc_m3, self.Lc_p1, self.Lc_p2, self.Lc_p3 \
            = vec.Lc_d0, vec.Lc_m1, vec.Lc_m2, vec.Lc_m3, vec.Lc_p1, vec.Lc_p2, vec.Lc_p3
        self.Lp_m2, self.Lp_m1, self.Lp_d0, self.Lp_p1, self.Lp_p2, self.Lp_p3 \
            = vec.Lp_m2, vec.Lp_m1, vec.Lp_d0, vec.Lp_p1, vec.Lp_p2, vec.Lp_p3
        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(size, sweeps)
        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Rci, self.Rcj = idx.Rci, idx.Rcj
        self.Lci, self.Lcj = idx.Lci, idx.Lcj
        self.Lpi, self.Lpj = idx.Lpi, idx.Lpj
        self.Li, self.Lj = idx.Li, idx.Lj

        # Flux jacobian data container
        A_data = np.concatenate((self.A_m3, self.A_m2, self.A_m1,       # Subdiagonals
                                 self.A_d0,                             # Diagonal
                                 self.A_p1, self.A_p2))                 # Superdiagonals
        # Right eigenvectors data container
        Rc_data = np.concatenate((self.Rc_m3, self.Rc_m2, self.Rc_m1,   # Subdiagonals
                                  self.Rc_d0,                           # Diagonal
                                  self.Rc_p1, self.Rc_p2, self.Rc_p3))  # Superdiagonals
        # Left eigenvectors data container
        Lc_data = np.concatenate((self.Lc_m3, self.Lc_m2, self.Lc_m1,   # Subdiagonals
                                  self.Lc_d0,                           # Diagonal
                                  self.Lc_p1, self.Lc_p2, self.Lc_p3))  # Superdiagonals
        # Left eigenvectors primitive data container
        Lp_data = np.concatenate((self.Lp_m2, self.Lp_m1,               # Subdiagonals
                                  self.Lp_d0,                           # Diagonal
                                  self.Lp_p1, self.Lp_p2, self.Lp_p3))  # Superdiagonals
        # Eigenvalue data container
        L_data = np.zeros((4 * (size + 1) * sweeps), dtype=float)

        # Build sparse matrices
        self.Jac    = coo((A_data, (self.Ai, self.Aj)))
        self.Rc     = coo((Rc_data, (self.Rci, self.Rcj)))
        self.Lc     = coo((Lc_data, (self.Lci, self.Lcj)))
        self.Lp     = coo((Lp_data, (self.Lpi, self.Lpj)))
        self.Lambda = coo((L_data, (self.Li, self.Lj)))

    def compute_flux_jacobian(self,
                              Wroe: RoePrimitiveState,
                              uv: np.ndarray = None,
                              ghv: np.ndarray = None,
                              ) -> None:
        """
        Calculate the x-direction sparse flux jacobian matrix A. A is defined as:

        \mathcal{A} = \left[\begin{array}{cccc}
                      0 & 1 & 0 & 0\\
                      \frac{\gamma-1}{2}\mathbf{V}^2-u^2 & -u\,\left(\gamma-3\right) & -v\,\left(\gamma-1\right) & \gamma-1\\
                      -u\,v & v & u & 0\\
                      u\left(\frac{\gamma-1}{2}\mathbf{V}^2 - H\right) & H + (1-\gamma)u^2 & -u\,v\,\left(\gamma-1\right) & \gamma u \end{array}\right]

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
        u2 = Wroe.u ** 2
        ghek = self.gh * Ek

        _sz = (self.size + 1) * self.sweeps

        # -3 subdiagonal entries
        self.Jac.data[0  * _sz:1  * _sz] = Wroe.u * (ghek - H)
        # -2 subdiagonal entries
        self.Jac.data[1  * _sz:2  * _sz] = -uv
        self.Jac.data[2  * _sz:3  * _sz] = H - self.gh * u2
        # -1 subdiagonal entries
        self.Jac.data[3  * _sz:4  * _sz] = ghek - u2
        self.Jac.data[4  * _sz:5  * _sz] = Wroe.v
        self.Jac.data[5  * _sz:6  * _sz] = -self.gh * uv
        # diagonal entries
        self.Jac.data[6  * _sz:7  * _sz] = self.gb * Wroe.u
        self.Jac.data[7  * _sz:8  * _sz] = Wroe.u
        self.Jac.data[8  * _sz:9  * _sz] = self.g * Wroe.u
        # +1 subdiagonal entries
        self.Jac.data[10 * _sz:11 * _sz] = -ghv


    def compute_Rc(self,
                   Wroe: RoePrimitiveState,
                   Lm: np.ndarray,
                   Lp: np.ndarray,
                   ua: np.ndarray = None,
                   ) -> None:

        H   = Wroe.H()
        Ek  = Wroe.Ek()
        ua  = Wroe.u * Wroe.a() if ua is None else ua

        _sz = (self.size + 1) * self.sweeps

        # -3 subdiagonal entries
        self.Rc.data[0 * _sz:1 * _sz]   = H - ua
        # -2 subdiagonal entries
        self.Rc.data[1 * _sz:2 * _sz]   = Wroe.v
        self.Rc.data[2 * _sz:3 * _sz]   = Ek
        # -1 subdiagonal entries
        self.Rc.data[3 * _sz:4 * _sz]   = Lm
        self.Rc.data[4 * _sz:5 * _sz]   = Wroe.v
        self.Rc.data[5 * _sz:6 * _sz]   = Wroe.v
        # diagonal entries
        self.Rc.data[7 * _sz:8 * _sz]   = Wroe.u
        self.Rc.data[9 * _sz:10 * _sz]  = H + ua
        # +1 subdiagonal entries
        self.Rc.data[11 * _sz:12 * _sz] = Wroe.v
        # +2 subdiagonal entries
        self.Rc.data[12 * _sz:13 * _sz] = Lp


    def compute_Lc(self,
                   Wroe: RoePrimitiveState,
                   ua: np.ndarray = None,
                   ) -> None:

        # Compute flow variables
        a       = Wroe.a()
        Ek      = Wroe.Ek()
        ua      = Wroe.u * Wroe.a() if ua is None else ua
        a2      = a ** 2
        ta2     = a2 * 2
        ghek    = self.gh * Ek
        gtu     = self.gt * Wroe.u
        gtv     = self.gt * Wroe.v

        _sz = (self.size + 1) * self.sweeps

        # -3 subdiagonal entries
        self.Lc.data[0  * _sz:1  * _sz] = -Wroe.v
        # -2 subdiagonal entries
        self.Lc.data[1  * _sz:2  * _sz] = (ghek - ua) / ta2
        # -1 subdiagonal entries
        self.Lc.data[2  * _sz:3  * _sz] = (a2 - ghek) / a2
        self.Lc.data[3 * _sz:4 * _sz]   = (gtu + a) / ta2
        # diagonal entries
        self.Lc.data[5  * _sz:6  * _sz] = (ghek + ua) / ta2
        self.Lc.data[6 * _sz:7 * _sz]   = self.gh * Wroe.u / a2
        self.Lc.data[7 * _sz:8 * _sz]   = gtv / ta2
        # +1 subdiagonal entries
        self.Lc.data[8  * _sz:9 * _sz]  = (gtu - a) / ta2
        self.Lc.data[9 * _sz:10 * _sz]  = self.gh * Wroe.v / a2
        self.Lc.data[10 * _sz:11 * _sz] = self.gh / ta2
        # +2 subdiagonal entries
        self.Lc.data[11 * _sz:12 * _sz] = gtv / ta2
        self.Lc.data[12 * _sz:13 * _sz] = self.gt / a2
        # +3 subdiagonal entries
        self.Lc.data[13 * _sz:14 * _sz] = self.gh / ta2


    def compute_Lp(self,
                   Wroe: RoePrimitiveState,
                   ) -> None:

        a = Wroe.a()
        r2a = Wroe.rho / (2 * a)
        one_a2 = 1 / (a ** 2)

        _sz = (self.size + 1) * self.sweeps

        # -2 subdiagonal entries
        self.Lp.data[0 * _sz:1 * _sz] = r2a
        # -1 subdiagonal entries
        self.Lp.data[3 * _sz:4 * _sz] = 0.5 * one_a2
        # +1 subdiagonal entries
        self.Lp.data[4 * _sz:5 * _sz] = -r2a
        # +2 subdiagonal entries
        self.Lp.data[5 * _sz:6 * _sz] = -one_a2
        # +3 subdiagonal entries
        self.Lp.data[6 * _sz:7 * _sz] = 0.5 * one_a2


    def compute_eigenvalues(self,
                            Wroe: RoePrimitiveState,
                            Lp: np.ndarray,
                            Lm: np.ndarray) -> None:

        _sz = (self.size + 1) * self.sweeps

        self.Lambda.data[0  * _sz:1  * _sz] = Lm
        self.Lambda.data[1  * _sz:2  * _sz] = Wroe.u
        self.Lambda.data[2  * _sz:3  * _sz] = Wroe.u
        self.Lambda.data[3  * _sz:4  * _sz] = Lp


    def compute_flux(self,
                     WL: PrimitiveState,
                     WR: PrimitiveState,
                     UL: ConservativeState,
                     UR: ConservativeState,
                     ) -> np.ndarray:
        """
        Computes the flux using the Roe approximate riemann solver. First, the Roe average state is computed based on
        the given left and right states, and then, the euler eigensystem is computed by diagonalization. The
        eigensystem is then used to compute the flux via the Roe flux function.

        Diagonalization
        ---------------

        The diagonalizaion is possible due to the hyperbolicty of the Euler equations. The 2D Euler equation in
        conservation form is given as:

        dU     dF     dG
        --  +  --  +  --  =  0
        dt     dx     dy

        .. math:
            \partial_t \mathbf{U} + \partial_x \mathbf{F} + \partial_y \mathbf{G} = 0,

        where :math:'U' is the vector of conserved variables, and :math:'F' and :math:'G' are the x and y direction
        fluxes, respectively. To diagonalize the system, the Euler equations can be re-expressed as:

        dU     dF dU     dG dU     dU      dU      dU
        --  +  -- --  +  -- --  =  --  +  A--  +  B--
        dt     dU dx     dU dy     dt      dx      dy

        .. math:
            \partial_t \mathbf{U} + \mathcal{A}\partial_x \mathbf{U} + \mathcal{B}\partial_y \mathbf{U} = 0,

        where :math:'A' and :math:'B' are the x and y direction flux jacobians, respectively.

        to be continued...

        """
        Wroe = RoePrimitiveState(self.inputs, WL, WR)
        if self.inputs.upwind_mode == 'conservative':
            return 0.5 * (WL.F() + WR.F()) - self.get_upwind_flux_conservative(Wroe, WL, WR, UL, UR)
        elif self.inputs.upwind_mode == 'primitive':
            return 0.5 * (WL.F() + WR.F()) - self.get_upwind_flux_primitive(Wroe, WL, WR)
        else:
            raise ValueError('Must specify type of upwinding')

    def diagonalize_primitive(self,
                              Wroe: RoePrimitiveState,
                              WL: PrimitiveState,
                              WR: PrimitiveState
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


    def diagonalize_conservative(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState) -> None:
        """
        Diagonalization of the Euler equations in conservative form. First, the Harten correction is applied to prevent
        expansion shocks, and then, the right-conservative, left-conservative primitive eigenvectors, and eigenvalues
        are calculated.

        Parameters:
            - Wroe (RoePrimitiveState): Primitive state holding the Roe averages
            - WL (PrimitiveState): Primitive left states
            - WR (PrimitiveState): Primitive right states

        Returns:
            - None
        """
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)
        self.compute_Rc(Wroe, Lm, Lp)
        self.compute_Lc(Wroe)
        self.compute_eigenvalues(Wroe, Lp, Lm)


    def get_upwind_flux_conservative(self,
                                     Wroe: RoePrimitiveState,
                                     WL: PrimitiveState,
                                     WR: PrimitiveState,
                                     UL: ConservativeState,
                                     UR: ConservativeState,
                                     ) -> np.ndarray:

        self.diagonalize_conservative(Wroe, WL, WR)
        self.Lambda.data = np.absolute(self.Lambda.data)

        # Dissipative upwind term
        return 0.5 * (self.Rc *
                      (self.Lambda *
                       (self.Lc * (UR - UL).flatten()))
                      ).reshape(1, -1, 4)


    def get_upwind_flux_primitive(self,
                                  Wroe: RoePrimitiveState,
                                  WL: PrimitiveState,
                                  WR: PrimitiveState,
                                  ) -> np.ndarray:

        self.diagonalize_primitive(Wroe, WL, WR)
        self.Lambda.data = np.absolute(self.Lambda.data)

        # Dissipative upwind term
        return 0.5 * (self.Rc *
                      (self.Lambda *
                       (self.Lp * (WR - WL).flatten()))
                      ).reshape(1, -1, 4)
