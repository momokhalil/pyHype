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
from scipy.sparse import coo_matrix as coo
from pyHype.flux.base import FluxFunction
from pyHype.states.states import RoePrimitiveState, ConservativeState, PrimitiveState
from pyHype.flux.eigen_system import XDIR_EIGENSYSTEM_INDICES, XDIR_EIGENSYSTEM_VECTORS

from profilehooks import profile


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

        self.size = size



    def compute_flux_jacobian(self,
                              Wroe: RoePrimitiveState,
                              Ek: np.ndarray = None,
                              H: np.ndarray = None,
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
            - ghek (np.ndarray): gamma_hat * ek, where ek is the kinetic energy = 0.5 * (v^2 + u^2)
            - H (np.ndarray): Total enthalpy
            - uv (np.ndarray): u velocity * v velocity
            - u2 (np.ndarray): u velocity squared
            - ghv (np.ndarray): gamma_hat * v velocity

        Returns:
            - N.A
        """

        if not ghv:
            ghv = self.gh * Wroe.v

        u2 = Wroe.u ** 2

        if not uv:
            uv = Wroe.u * Wroe.v

        if Ek is None:
            ghek = self.gh * 0.5 * (Wroe.u ** 2 + Wroe.v ** 2)
        else:
            ghek = self.gh * Ek

        if H is None:
            H = Wroe.H()

        _sz = self.size + 1

        self.A.data[0 * _sz:1  * _sz] = Wroe.u * (ghek - H)
        self.A.data[1 * _sz:2  * _sz] = -uv
        self.A.data[2 * _sz:3 * _sz] = H - self.gh * u2
        self.A.data[3 * _sz:4  * _sz] = ghek - u2
        self.A.data[4 * _sz:5 * _sz] = Wroe.v
        self.A.data[5 * _sz:6 * _sz] = -self.gh * uv
        self.A.data[6 * _sz:7  * _sz] = self.gb * Wroe.u
        self.A.data[7 * _sz:8 * _sz] = Wroe.u
        self.A.data[8 * _sz:9 * _sz] = self.g * Wroe.u
        self.A.data[10 * _sz:11 * _sz] = -ghv


    def compute_Rc(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   H: np.ndarray,
                   Lm: np.ndarray,
                   Lp: np.ndarray,
                   Ek: np.ndarray = None,
                   ua: np.ndarray = None,
                   ) -> None:

        if ua is None:
            ua = Wroe.u * a

        if Ek is None:
            Ek = 0.5 * (Wroe.u ** 2 + Wroe.v ** 2)

        _sz = self.size + 1

        self.Rc.data[0 * _sz:1 * _sz] = H - ua
        self.Rc.data[1 * _sz:2 * _sz] = Wroe.v
        self.Rc.data[2 * _sz:3 * _sz] = Ek
        self.Rc.data[3 * _sz:4 * _sz] = Lm
        self.Rc.data[4 * _sz:5 * _sz] = Wroe.v
        self.Rc.data[5 * _sz:6 * _sz] = H + ua
        self.Rc.data[7 * _sz:8 * _sz] = Wroe.u
        self.Rc.data[8 * _sz:9 * _sz] = Wroe.v
        self.Rc.data[9 * _sz:10 * _sz] = Wroe.v

        self.Rc.data[11 * _sz:12 * _sz] = Lp


    def compute_Lc(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   Ek: np.ndarray = None,
                   ua: np.ndarray = None,
                   ) -> None:

        if Ek is None:
            ghek = self.gh * 0.5 * (Wroe.u ** 2 + Wroe.v ** 2)
        else:
            ghek = self.gh * Ek

        if ua is None:
            ua = Wroe.u * a

        a2 = a ** 2
        ta2 = a2 * 2
        gtu = self.gt * Wroe.u
        gtv = self.gt * Wroe.v
        ghv = self.gh * Wroe.v

        _sz = self.size + 1

        self.Lc.data[0  * _sz:1  * _sz] = -Wroe.v
        self.Lc.data[1  * _sz:2  * _sz] = (ghek - ua) / ta2
        self.Lc.data[2  * _sz:3  * _sz] = (a2 - ghek) / a2
        self.Lc.data[3 * _sz:4 * _sz] = (gtu + a) / ta2
        self.Lc.data[5  * _sz:6  * _sz] = (ghek + ua) / ta2
        self.Lc.data[6 * _sz:7 * _sz] = self.gh * Wroe.u / a2
        self.Lc.data[7 * _sz:8 * _sz] = gtv / ta2
        self.Lc.data[8  * _sz:9 * _sz] = (gtu - a) / ta2
        self.Lc.data[9 * _sz:10 * _sz] = ghv / a2
        self.Lc.data[10 * _sz:11 * _sz] = self.gh / ta2
        self.Lc.data[11 * _sz:12 * _sz] = gtv / ta2
        self.Lc.data[12 * _sz:13 * _sz] = self.gt / a2
        self.Lc.data[13 * _sz:14 * _sz] = self.gh / ta2


    def compute_Lp(self,
                   Wroe: RoePrimitiveState,
                   a: np.ndarray,
                   ) -> None:

        ap2 = Wroe.rho * a / 2
        _sz = self.size + 1

        self.Lp.data[1 * _sz:2 * _sz] = ap2
        self.Lp.data[3 * _sz:4 * _sz] = -ap2
        self.Lp.data[5 * _sz:6 * _sz] = -1 / (a ** 2)


    def compute_eigenvalues(self,
                            Wroe: RoePrimitiveState,
                            Lp: np.ndarray,
                            Lm: np.ndarray):

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.u
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.u

        self.Lambda.data = self.lam

    def diagonalize_primitive(self, Wroe, WL, WR):

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Speed of sound
        a = Wroe.a()
        # Kinetic Energy
        Ek = Wroe.Ek()
        # Enthalpy
        H = Wroe.H(Ek)

        # Flux Jacobian
        self.compute_flux_jacobian(Wroe, Ek, H)
        # Right eigenvectors
        self.compute_Rc(Wroe, a, H, Lm, Lp, Ek=Ek)
        # Left eigenvectors
        self.compute_Lp(Wroe, a)
        # Eigenvalues
        self.compute_eigenvalues(Wroe, Lp, Lm)


    def diagonalize_conservative(self, Wroe, WL, WR):

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Speed of sound
        a = Wroe.a()
        # Kinetic Energy
        Ek = Wroe.Ek()
        # Enthalpy
        H = Wroe.H(Ek)
        # u times a
        ua = Wroe.u * a

        # Flux Jacobian
        self.compute_flux_jacobian(Wroe, Ek, H)
        # Right eigenvectors
        self.compute_Rc(Wroe, a, H, Lm, Lp, Ek=Ek, ua=ua)
        # Left eigenvectors
        self.compute_Lc(Wroe, a, Ek=Ek, ua=ua)
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
        UpL = (UL + UR).flatten()
        nondis = 0.5 * self.A.dot(UpL).reshape(1, -1, 4)

        # Compute dissipative upwind flux term
        if self.inputs.upwind_mode == 'conservative':
            upwind = self.get_upwind_conservative(Wroe, WL, WR, UL, UR)
        elif self.inputs.upwind_mode == 'primitive':
            upwind = self.get_upwind_primitive(Wroe, WL, WR)
        else:
            raise ValueError('Must specify type of upwinding')

        return nondis - upwind


    def get_upwind_conservative(self,
                                Wroe: RoePrimitiveState,
                                WL: PrimitiveState,
                                WR: PrimitiveState,
                                UL: ConservativeState = None,
                                UR: ConservativeState = None,
                                ) -> np.ndarray:
        # Diagonalize
        self.diagonalize_conservative(Wroe, WL, WR)
        # Conservative state jump
        dU = (UR - UL).flatten()
        # absolute value
        self.Lambda.data = np.absolute(self.Lambda.data)

        # Dissipative upwind term
        return 0.5 * (self.Rc.dot(self.Lambda.dot(self.Lc.dot(dU)))).reshape(1, -1, 4)


    def get_upwind_primitive(self,
                             Wroe: RoePrimitiveState,
                             WL: PrimitiveState,
                             WR: PrimitiveState,
                             ) -> np.ndarray:
        # Diagonalize
        self.diagonalize_primitive(Wroe, WL, WR)
        # Primitive state jump
        dW = (WR - WL).flatten()
        # absolute value
        self.Lambda.data = np.absolute(self.Lambda.data)

        # Dissipative upwind term
        return 0.5 * (self.Rc.dot(self.Lambda.dot(self.Lp.dot(dW)))).reshape(1, -1, 4)
