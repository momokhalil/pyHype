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

import time
import numpy as np
from pyHype.utils.math.coo import coo
from pyHype.flux.base import FluxFunction
from pyHype.states.states import RoePrimitiveState, ConservativeState, PrimitiveState
from pyHype.flux.eigen_system import XDIR_EIGENSYSTEM_INDICES, XDIR_EIGENSYSTEM_VECTORS


class ROE_FLUX_X(FluxFunction):

    def __init__(self, inputs, size):

        # Superclass constructor
        super().__init__(inputs)

        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self.inputs, size)

        self.A_d0, self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 \
            = vec.A_d0, vec.A_m1, vec.A_m2, vec.A_m3, vec.A_p1, vec.A_p2

        self.Rc_d0, self.Rc_m1, self.Rc_m2, self.Rc_m3, self.Rc_p1, self.Rc_p2 \
            = vec.Rc_d0, vec.Rc_m1, vec.Rc_m2, vec.Rc_m3, vec.Rc_p1, vec.Rc_p2

        self.Lc_d0, self.Lc_m1, self.Lc_m2, self.Lc_m3, self.Lc_p1, self.Lc_p2, self.Lc_p3 \
            = vec.Lc_d0, vec.Lc_m1, vec.Lc_m2, vec.Lc_m3, vec.Lc_p1, vec.Lc_p2, vec.Lc_p3

        self.Lp_m1, self.Lp_p1, self.Lp_p2, self.Lp_p3 \
            = vec.Lp_m1, vec.Lp_p1, vec.Lp_p2, vec.Lp_p3

        self.lam = vec.lam

        # Rc-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(size)

        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Rci, self.Rcj = idx.Rci, idx.Rcj
        self.Lci, self.Lcj = idx.Lci, idx.Lcj
        self.Lpi, self.Lpj = idx.Lpi, idx.Lpj
        self.Li, self.Lj = idx.Li, idx.Lj

        # Build sparse matrices

        # Flux jacobian
        A_data = np.concatenate((self.A_m3, self.A_m2, self.A_m1,  # Subdiagonals
                                 self.A_d0,  # Diagonal
                                 self.A_p1, self.A_p2))  # Superdiagonals
        self.A = coo((A_data, (self.Ai, self.Aj)))

        # Right eigenvectors
        Rc_data = np.concatenate((self.Rc_m3, self.Rc_m2, self.Rc_m1,  # Subdiagonals
                                  self.Rc_d0,  # Diagonal
                                  self.Rc_p1, self.Rc_p2))  # Superdiagonals
        self.Rc = coo((Rc_data, (self.Rci, self.Rcj)))

        # Left eigenvectors
        Lc_data = np.concatenate((self.Lc_m3, self.Lc_m2, self.Lc_m1,  # Subdiagonals
                                  self.Lc_d0,  # Diagonal
                                  self.Lc_p1, self.Lc_p2, self.Lc_p3))  # Superdiagonals
        self.Lc = coo((Lc_data, (self.Lci, self.Lcj)))

        # Left eigenvectors primitive
        Lp_data = np.concatenate((self.Lp_m1,  # Subdiagonals
                                  self.Lp_p1, self.Lp_p2, self.Lp_p3))  # Superdiagonals
        self.Lp = coo((Lp_data, (self.Lpi, self.Lpj)))

        # Eigenvalues
        L_data = np.zeros((4 * size + 4))
        self.Lambda = coo((L_data, (self.Li, self.Lj)))

        # Set upwinding method
        if self.inputs.upwind_mode == 'conservative':
            self.upwind = self.get_upwind_conservative
        elif self.inputs.upwind_mode == 'primitive':
            self.upwind = self.get_upwind_primitive
        else:
            raise ValueError('Must specify type of upwinding')

    def compute_flux_jacobian(self,
                              Wroe: RoePrimitiveState,
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
            - ghek (np.ndarray): gamma_hat * ek, where ek is the kinetic energy = 0.5 * (v^2 + u^2)
            - H (np.ndarray): Total enthalpy
            - uv (np.ndarray): u velocity * v velocity
            - u2 (np.ndarray): u velocity squared
            - ghv (np.ndarray): gamma_hat * v velocity

        Returns:
            - N.A
        """

        ghv = self.gh * Wroe.v
        u2 = Wroe.u ** 2
        uv = Wroe.u * Wroe.v
        ghek = self.gh * 0.5 * (Wroe.u ** 2 + Wroe.v ** 2)
        H = Wroe.H()

        self.A_m3[:] = Wroe.u * (ghek - H)
        self.A_m2[0::2] = -uv
        self.A_m2[1::2] = H - self.gh * u2
        self.A_m1[0::3] = ghek - u2
        self.A_m1[1::3] = Wroe.v
        self.A_m1[2::3] = -self.gh * uv
        self.A_d0[0::3] = self.gb * Wroe.u
        self.A_d0[1::3] = Wroe.u
        self.A_d0[2::3] = self.g * Wroe.u
        self.A_p1[1::2] = -ghv

        data = np.concatenate((self.A_m3, self.A_m2, self.A_m1,
                                      self.A_d0,
                                      self.A_p1, self.A_p2),
                                     axis=0)

        self.A = coo((data, (self.Ai, self.Aj))).eliminate_zeros()

    def compute_Rc(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   H: np.ndarray,
                   Lm: np.ndarray,
                   Lp: np.ndarray
                   ) -> None:

        ua = Wroe.u * a
        ek = 0.5 * (Wroe.u ** 2 + Wroe.v ** 2)

        self.Rc_m3[:] = H - ua
        self.Rc_m2[0::2] = Wroe.v
        self.Rc_m2[1::2] = ek
        self.Rc_m1[0::3] = Lm
        self.Rc_m1[1::3] = Wroe.v
        self.Rc_m1[2::3] = H + ua
        self.Rc_d0[1::4] = Wroe.u
        self.Rc_d0[2::4] = Wroe.v
        self.Rc_d0[3::4] = Wroe.v
        self.Rc_p1[1::3] = Lp

        data = np.concatenate((self.Rc_m3, self.Rc_m2, self.Rc_m1,
                                       self.Rc_d0,
                                       self.Rc_p1, self.Rc_p2),
                                      axis=0)
        self.Rc = coo((data, (self.Rci, self.Rcj)))
        self.Rc.eliminate_zeros()

    def compute_Lc(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   ) -> None:

        ghek = 0.5 * self.gh * (Wroe.u ** 2 + Wroe.v ** 2)
        ua = Wroe.u * a
        a2 = a ** 2
        ta2 = a2 * 2
        gtu = self.gt * Wroe.u
        gtv = self.gt * Wroe.v
        ghv = self.gh * Wroe.v

        self.Lc_m3[:] = -Wroe.v
        self.Lc_m2[:] = (ghek - ua) / ta2
        self.Lc_m1[0::3] = (a2 - ghek) / a2
        self.Lc_m1[1::3] = (gtu + a) / ta2
        self.Lc_d0[0::3] = (ghek + ua) / ta2
        self.Lc_d0[1::3] = self.gh * Wroe.u / a2
        self.Lc_d0[2::3] = gtv / ta2
        self.Lc_p1[0::3] = (gtu - a) / ta2
        self.Lc_p1[1::3] = ghv / a2
        self.Lc_p1[2::3] = self.gh / ta2
        self.Lc_p2[0::2] = gtv / ta2
        self.Lc_p2[1::2] = self.gt / a2
        self.Lc_p3[:] = self.gh / ta2

        data = np.concatenate((self.Lc_m3, self.Lc_m2, self.Lc_m1,
                                       self.Lc_d0,
                                       self.Lc_p1, self.Lc_p2, self.Lc_p3),
                                      axis=0)

        self.Lc = coo((data, (self.Lci, self.Lcj)))
        self.Lc.eliminate_zeros()

    def compute_Lp(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   ) -> None:

        ap2 = Wroe.rho * a / 2

        self.Lp_m1[1::3] = ap2
        self.Lp_p1[0::2] = -ap2
        self.Lp_p2[:] = -1 / (a ** 2)

        data = np.concatenate((self.Lp_m1,
                                       self.Lp_p1, self.Lp_p2, self.Lp_p3),
                                      axis=0)

        self.Lp = coo((data, (self.Lpi, self.Lpj)))
        self.Lp.eliminate_zeros()

    def compute_eigenvalues(self,
                            Wroe: RoePrimitiveState,
                            Lp: np.ndarray,
                            Lm: np.ndarray):

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.u
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.u

        data = self.lam

        self.Lambda = coo((data, (self.Li, self.Lj)))
        self.Lambda.eliminate_zeros()

    def diagonalize(self, Wroe, WL, WR):

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Speed of sound
        a = Wroe.a()
        # Enthalpy
        H = Wroe.H()

        # Right eigenvectors
        self.compute_Rc(Wroe, a, H, Lm, Lp)
        # Left eigenvectors
        self.compute_Lp(Wroe, a)
        # Eigenvalues
        self.compute_eigenvalues(Wroe, Lp, Lm)

    def diagonalize_con(self, Wroe, WL, WR):

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Speed of sound
        a = Wroe.a()
        # Enthalpy
        H = Wroe.H()

        # Right eigenvectors
        self.compute_Rc(Wroe, a, H, Lm, Lp)
        # Left eigenvectors
        self.compute_Lc(Wroe, a)
        # Eigenvalues
        self.compute_eigenvalues(Wroe, Lp, Lm)

    def compute_flux(self,
                     UL: ConservativeState,
                     UR: ConservativeState
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

        # Create Left and Right PrimitiveStates
        WL = UL.to_primitive_state()
        WR = UR.to_primitive_state()

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, WL, WR)

        # Compute non-dissipative flux term
        nondis = 0.5 * (UL.F() + UR.F())

        # Compute dissipative upwind flux term
        upwind = self.upwind(Wroe, WL, WR, UL, UR)

        return nondis - upwind


    def get_upwind_conservative(self,
                                Wroe: RoePrimitiveState,
                                WL: PrimitiveState,
                                WR: PrimitiveState,
                                UL: ConservativeState = None,
                                UR: ConservativeState = None,
                                ) -> np.ndarray:
        # Diagonalize
        self.diagonalize_con(Wroe, WL, WR)
        # Conservative state jump
        dU = (UR - UL).flatten()
        # absolute value
        absL = np.absolute(self.Lambda)
        # Prune
        absL.eliminate_zeros()

        # Dissipative upwind term
        return (self.Rc @ absL @ self.Lc @ dU).reshape(1, -1, 4)


    def get_upwind_primitive(self,
                             Wroe: RoePrimitiveState,
                             WL: PrimitiveState,
                             WR: PrimitiveState,
                             UL: ConservativeState = None,
                             UR: ConservativeState = None,
                             ) -> np.ndarray:
        # Diagonalize
        self.diagonalize(Wroe, WL, WR)
        # Primitive state jump
        dW = (WR - WL).flatten()
        # absolute value
        absL = np.absolute(self.Lambda)
        # Prune
        absL.eliminate_zeros()

        # Dissipative upwind term
        return 0.5 * (self.Rc @ absL @ self.Lp @ dW).reshape(1, -1, 4)
