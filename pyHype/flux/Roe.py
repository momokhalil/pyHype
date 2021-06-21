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
import scipy.sparse as sparse
from pyHype.flux.base import FluxFunction
from pyHype.states.states import RoePrimitiveState
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

        self.lam = vec.lam

        # Rc-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(size)

        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Rci, self.Rcj = idx.Rci, idx.Rcj
        self.Lci, self.Lcj = idx.Lci, idx.Lcj
        self.Li, self.Lj = idx.Li, idx.Lj

        # Build sparse matrices

        # Flux jacobian
        A_data = np.concatenate((self.A_m3, self.A_m2, self.A_m1,       # Subdiagonals
                                 self.A_d0,                             # Diagonal
                                 self.A_p1, self.A_p2))                 # Superdiagonals
        self.A = sparse.coo_matrix((A_data, (self.Ai, self.Aj)))

        # Right eigenvectors
        Rc_data = np.concatenate((self.Rc_m3, self.Rc_m2, self.Rc_m1,   # Subdiagonals
                                  self.Rc_d0,                           # Diagonal
                                  self.Rc_p1, self.Rc_p2))              # Superdiagonals
        self.Rc = sparse.coo_matrix((Rc_data, (self.Rci, self.Rcj)))

        # Left eigenvectors
        Lc_data = np.concatenate((self.Lc_m3, self.Lc_m2, self.Lc_m1,   # Subdiagonals
                                  self.Lc_d0,                           # Diagonal
                                  self.Lc_p1, self.Lc_p2, self.Lc_p3))  # Superdiagonals
        self.Lc = sparse.coo_matrix((Lc_data, (self.Lci, self.Lcj)))

        # Eigenvalues
        L_data = np.zeros((4 * size + 4))
        self.Lambda = sparse.coo_matrix((L_data, (self.Li, self.Lj)))


    def compute_flux_jacobian(self,
                              Wroe: RoePrimitiveState,
                              ghek: np.ndarray,
                              H: np.ndarray,
                              uv: np.ndarray,
                              u2: np.ndarray,
                              ghv: np.ndarray
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

        # Flux Jacobian
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

        self.A.data = np.concatenate((self.A_m3, self.A_m2, self.A_m1,
                                      self.A_d0,
                                      self.A_p1, self.A_p2),
                                     axis=0)


    def compute_right_eigenvectors(self,
                                   Wroe: RoePrimitiveState,
                                   H: np.ndarray,
                                   ua: np.ndarray,
                                   ek: np.ndarray,
                                   Lm: np.ndarray,
                                   Lp: np.ndarray
                                   ) -> None:

        # right Eigenvectors
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

        self.Rc.data = np.concatenate((self.Rc_m3, self.Rc_m2, self.Rc_m1,
                                      self.Rc_d0,
                                      self.Rc_p1, self.Rc_p2),
                                      axis=0)


    def compute_left_eigenvectors(self,
                                  Wroe: RoePrimitiveState,
                                  a: np.ndarray,
                                  ghek: np.ndarray,
                                  ua: np.ndarray,
                                  ta2: np.ndarray,
                                  a2: np.ndarray,
                                  gtu: np.ndarray,
                                  gtv: np.ndarray,
                                  ghv: np.ndarray
                                  ) -> None:

        # left eigenvectors
        self.Lc_m3[:] = Wroe.v
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

        self.Lc.data = np.concatenate((self.Lc_m3, self.Lc_m2, self.Lc_m1,
                                       self.Lc_d0,
                                       self.Lc_p1, self.Lc_p2, self.Lc_p3),
                                      axis=0)


    def compute_eigenvalues(self,
                            Wroe: RoePrimitiveState,
                            Lp: np.ndarray,
                            Lm: np.ndarray):

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.u
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.u

        self.Lambda.data = self.lam


    def diagonalize(self, Wroe, WL, WR):

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Calculate quantities to construct eigensystem
        a       = Wroe.a()
        H       = Wroe.H()
        gtu     = self.gt * Wroe.u
        ghv     = self.gh * Wroe.v
        gtv     = self.gt * Wroe.v
        u2      = Wroe.u ** 2
        uv      = Wroe.u * Wroe.v
        ek      = 0.5 * (u2 + Wroe.v ** 2)
        a2      = a ** 2
        ta2     = a2 * 2
        ua      = Wroe.u * a
        ghek    = self.gh * ek

        # Flux Jacobian
        self.compute_flux_jacobian(Wroe, ghek, H, uv, u2, ghv)

        # Right eigenvectors
        self.compute_right_eigenvectors(Wroe, H, ua, ek, Lm, Lp)

        # Left eigenvectors
        self.compute_left_eigenvectors(Wroe, a, ghek, ua, ta2, a2, gtu, gtv, ghv)

        # Eigenvalues
        self.compute_eigenvalues(Wroe, Lp, Lm)


    def compute_flux(self, UL, UR):

        # Create Left and Right PrimitiveStates
        WL, WR = UL.to_primitive_state(), UR.to_primitive_state()
        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, WL, WR)
        # Get eigenstructure
        self.diagonalize(Wroe, WL, WR)

        # Left state plus right state
        LpR = (UL + UR).flatten()
        # Left state minus right state
        LmR = (UL - UR).flatten()
        # absolute value
        absL = np.absolute(self.Lambda)

        # Non-dissipative flux term
        nondisp = self.A.dot(LpR)
        # Dissipative upwind term
        upwind = self.Rc.dot(absL.dot(self.Lc.dot(LmR)))

        return 0.5 * (nondisp + upwind).reshape(1, -1, 4)
