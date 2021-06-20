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
        super().__init__(inputs)

        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self.inputs, size)

        self.A_d0 = vec.A_d0

        self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 \
            = vec.A_m1, vec.A_m2, vec.A_m3, vec.A_p1, vec.A_p2

        self.X_d0, self.X_m1, self.X_m2, self.X_m3, self.X_p1, self.X_p2 \
            = vec.X_d0, vec.X_m1, vec.X_m2, vec.X_m3, vec.X_p1, vec.X_p2

        self.Xi_d0, self.Xi_m1, self.Xi_m2, self.Xi_m3, self.Xi_p1, self.Xi_p2, self.Xi_p3 \
            = vec.Xi_d0, vec.Xi_m1, vec.Xi_m2, vec.Xi_m3, vec.Xi_p1, vec.Xi_p2, vec.Xi_p3

        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(size)

        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Xi, self.Xj = idx.Xi, idx.Xj
        self.Xi_i, self.Xi_j = idx.Xi_i, idx.Xi_j
        self.Li, self.Lj = idx.Li, idx.Lj

        # Build sparse matrices
        data = np.concatenate((self.A_m3, self.A_m2, self.A_m1, self.A_d0, self.A_p1, self.A_p2))
        self.A = sparse.coo_matrix((data, (self.Ai, self.Aj)))

        data = np.concatenate((self.X_m3, self.X_m2, self.X_m1, self.X_d0, self.X_p1, self.X_p2))
        self.X = sparse.coo_matrix((data, (self.Xi, self.Xj)))

        data = np.concatenate((self.Xi_m3, self.Xi_m2, self.Xi_m1, self.Xi_d0, self.Xi_p1, self.Xi_p2, self.Xi_p3))
        self.Xi = sparse.coo_matrix((data, (self.Xi_i, self.Xi_j)))

        data = np.zeros((4 * size + 4))
        self.Lambda = sparse.coo_matrix((data, (self.Li, self.Lj)))


    def _compute_flux_jacobian(self, Wroe, ghek, H, uv, u2, ghv):

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


    def _compute_left_eigenvectors(self, Wroe, H, ua, ek, Lm, Lp):

        # Left Eigenvectors
        self.X_m3[:] = H - ua
        self.X_m2[0::2] = Wroe.v
        self.X_m2[1::2] = ek
        self.X_m1[0::3] = Lm
        self.X_m1[1::3] = Wroe.v
        self.X_m1[2::3] = H + ua
        self.X_d0[1::4] = Wroe.u
        self.X_d0[2::4] = Wroe.v
        self.X_d0[3::4] = Wroe.v
        self.X_p1[1::3] = Lp

        self.X.data = np.concatenate((self.X_m3, self.X_m2, self.X_m1,
                                      self.X_d0,
                                      self.X_p1, self.X_p2),
                                     axis=0)


    def _compute_right_eigenvectors(self, Wroe, a, ghek, ua, ta2, a2, gtu, gtv, ghv):

        # Left eigenvectors
        self.Xi_m3[:] = Wroe.v
        self.Xi_m2[:] = (ghek - ua) / ta2
        self.Xi_m1[0::3] = (a2 - ghek) / a2
        self.Xi_m1[1::3] = (gtu + a) / ta2
        self.Xi_d0[0::3] = (ghek + ua) / ta2
        self.Xi_d0[1::3] = self.gh * Wroe.u / a2
        self.Xi_d0[2::3] = gtv / ta2
        self.Xi_p1[0::3] = (gtu - a) / ta2
        self.Xi_p1[1::3] = ghv / a2
        self.Xi_p1[2::3] = self.gh / ta2
        self.Xi_p2[0::2] = gtv / ta2
        self.Xi_p2[1::2] = self.gt / a2
        self.Xi_p3[:] = self.gh / ta2

        self.Xi.data = np.concatenate((self.Xi_m3, self.Xi_m2, self.Xi_m1,
                                       self.Xi_d0,
                                       self.Xi_p1, self.Xi_p2, self.Xi_p3),
                                      axis=0)


    def _compute_eigenvalues(self, Wroe, Lp, Lm):

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.u
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.u

        self.Lambda.data = self.lam


    def _get_eigen_system_from_roe_state(self, UL, UR):
        # Create Left and Right PrimitiveStates
        WL, WR = UL.to_primitive_state(), UR.to_primitive_state()

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, WL, WR)

        # Harten entropy correction
        Lm, Lp = self.harten_correction_x(Wroe, WL, WR)

        # Calculate quantities to construct eigensystem
        a = Wroe.a()
        H = Wroe.H()
        gtu = self.gt * Wroe.u
        ghv = self.gh * Wroe.v
        gtv = self.gt * Wroe.v
        u2 = Wroe.u ** 2
        uv = Wroe.u * Wroe.v
        ek = 0.5 * (u2 + Wroe.v ** 2)
        a2 = a ** 2
        ta2 = a2 * 2
        ua = Wroe.u * a
        ghek = self.gh * ek

        # Flux Jacobian
        self._compute_flux_jacobian(Wroe, ghek, H, uv, u2, ghv)

        # Left eigenvectors
        self._compute_left_eigenvectors(Wroe, H, ua, ek, Lm, Lp)

        # Right eigenvectors
        self._compute_right_eigenvectors(Wroe, a, ghek, ua, ta2, a2, gtu, gtv, ghv)

        # Eigenvalues
        self._compute_eigenvalues(Wroe, Lp, Lm)


    def get_flux(self, UL, UR):
        self._get_eigen_system_from_roe_state(UL, UR)

        return 0.5 * (self.A.dot((UL + UR).flatten()) +
                      self.X.dot(
                                np.absolute(self.Lambda).dot(
                                                            self.Xi.dot(
                                                                       (UL - UR).flatten()
                                                                       )
                                                            )
                                )
                      ).reshape(1, -1, 4)
