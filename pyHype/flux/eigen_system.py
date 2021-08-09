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


class XDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, inputs, nx):

        self.inputs = inputs

        self.A_d0 = np.ones((3 * (nx + 1)))
        self.A_m1 = np.ones((3 * (nx + 1)))
        self.A_m2 = np.ones((2 * (nx + 1)))
        self.A_m3 = np.ones((1 * (nx + 1)))
        self.A_p1 = np.ones((2 * (nx + 1)))
        self.A_p2 = np.full((1 * (nx + 1)), self.inputs.gamma - 1)

        self.Rc_d0 = np.ones((4 * (nx + 1)))
        self.Rc_m1 = np.ones((3 * (nx + 1)))
        self.Rc_m2 = np.ones((2 * (nx + 1)))
        self.Rc_m3 = np.ones((1 * (nx + 1)))
        self.Rc_p1 = np.ones((3 * (nx + 1)))
        self.Rc_p2 = np.ones((1 * (nx + 1)))

        self.Lc_d0 = np.ones((3 * (nx + 1)))
        self.Lc_m1 = np.ones((3 * (nx + 1)))
        self.Lc_m2 = np.ones((1 * (nx + 1)))
        self.Lc_m3 = np.ones((1 * (nx + 1)))
        self.Lc_p1 = np.ones((3 * (nx + 1)))
        self.Lc_p2 = np.ones((2 * (nx + 1)))
        self.Lc_p3 = np.ones((1 * (nx + 1)))

        self.Lp_m1 = np.ones((3 * (nx + 1)))
        self.Lp_p1 = 0.5 * np.ones((2 * (nx + 1)))
        self.Lp_p2 = np.ones((1 * (nx + 1)))
        self.Lp_p3 = 0.5 * np.ones((1 * (nx + 1)))

        self.lam = np.zeros((4 * (nx + 1)))


class XDIR_EIGENSYSTEM_INDICES:
    def __init__(self, nx):

        # Flux Jacobian indices
        self._get_flux_jacobian_indices(nx)
        # Right Eigenvector conservative formulation indices
        self._get_Rc_indices(nx)
        # Left Eigenvector conservative formulation indices
        self._get_Lc_indices(nx)
        # Left Eigenvector primitive formulation indices
        self._get_Lp_indices(nx)
        # Eigenvalues indices
        self._get_eigenvalue_indices(nx)

    def _get_flux_jacobian_indices(self, nx):
        """
        Build the i and j indices for the flux jacobian sparse matrix construction. The flux jacobian matrix for each
        cell has the sparsity pattern of:

        0   X   0   0
        X   X   X   X
        X   X   X   0
        X   X   X   X
                        0   X   0   0
                        X   X   X   X
                        X   X   X   0
                        X   X   X   X
                                        .
                                            .
                                                .
                                                    .
                                                        0   X   0   0
                                                        X   X   X   X
                                                        X   X   X   0
                                                        X   X   X   X
        """

        A_d0_i = self.get_indices(nx, [1, 2, 3])
        A_d0_j = self.get_indices(nx, [1, 2, 3])
        A_m1_i = self.get_indices(nx, [1, 2, 3])
        A_m1_j = self.get_indices(nx, [0, 1, 2])
        A_m2_i = self.get_indices(nx, [2, 3])
        A_m2_j = self.get_indices(nx, [0, 1])
        A_m3_i = self.get_indices(nx, [3])
        A_m3_j = self.get_indices(nx, [0])
        A_p1_i = self.get_indices(nx, [0, 1])
        A_p1_j = self.get_indices(nx, [1, 2])
        A_p2_i = self.get_indices(nx, [1])
        A_p2_j = self.get_indices(nx, [3])

        self.Ai = np.hstack((A_m3_i, A_m2_i, A_m1_i, A_d0_i, A_p1_i, A_p2_i))
        self.Aj = np.hstack((A_m3_j, A_m2_j, A_m1_j, A_d0_j, A_p1_j, A_p2_j))

    def _get_Rc_indices(self, nx):
        """
        Build the i and j indices for the Rc sparse matrix construction. The Rc matrix for each
        cell has the sparsity pattern of:

        X   X   X   0
        X   X   X   0
        X   X   X   X
        X   X   X   X
                        X   X   X   0
                        X   X   X   0
                        X   X   X   X
                        X   X   X   X
                                        .
                                            .
                                                .
                                                    .
                                                        X   X   X   0
                                                        X   X   X   0
                                                        X   X   X   X
                                                        X   X   X   X
        """

        Rc_d0_i = self.get_indices(nx, [0, 1, 2, 3])
        Rc_d0_j = self.get_indices(nx, [0, 1, 2, 3])
        Rc_m1_i = self.get_indices(nx, [1, 2, 3])
        Rc_m1_j = self.get_indices(nx, [0, 1, 2])
        Rc_m2_i = self.get_indices(nx, [2, 3])
        Rc_m2_j = self.get_indices(nx, [0, 1])
        Rc_m3_i = self.get_indices(nx, [3])
        Rc_m3_j = self.get_indices(nx, [0])
        Rc_p1_i = self.get_indices(nx, [0, 1, 2])
        Rc_p1_j = self.get_indices(nx, [1, 2, 3])
        Rc_p2_i = self.get_indices(nx, [0])
        Rc_p2_j = self.get_indices(nx, [2])

        self.Rci = np.hstack((Rc_m3_i, Rc_m2_i, Rc_m1_i, Rc_d0_i, Rc_p1_i, Rc_p2_i))
        self.Rcj = np.hstack((Rc_m3_j, Rc_m2_j, Rc_m1_j, Rc_d0_j, Rc_p1_j, Rc_p2_j))

    def _get_Lc_indices(self, nx):
        """
        Build the i and j indices for the Lc sparse matrix construction. The Lc matrix for each
        cell has the sparsity pattern of:

        X   X   X   X
        X   X   X   X
        X   X   X   X
        X   0   X   0
                        X   X   X   X
                        X   X   X   X
                        X   X   X   X
                        X   0   X   0
                                        .
                                            .
                                                .
                                                    .
                                                        X   X   X   X
                                                        X   X   X   X
                                                        X   X   X   X
                                                        X   0   X   0
        """

        Lc_d0_i = self.get_indices(nx, [0, 1, 2])
        Lc_d0_j = self.get_indices(nx, [0, 1, 2])
        Lc_m1_i = self.get_indices(nx, [1, 2, 3])
        Lc_m1_j = self.get_indices(nx, [0, 1, 2])
        Lc_m2_i = self.get_indices(nx, [2])
        Lc_m2_j = self.get_indices(nx, [0])
        Lc_m3_i = self.get_indices(nx, [3])
        Lc_m3_j = self.get_indices(nx, [0])
        Lc_p1_i = self.get_indices(nx, [0, 1, 2])
        Lc_p1_j = self.get_indices(nx, [1, 2, 3])
        Lc_p2_i = self.get_indices(nx, [0, 1])
        Lc_p2_j = self.get_indices(nx, [2, 3])
        Lc_p3_i = self.get_indices(nx, [0])
        Lc_p3_j = self.get_indices(nx, [3])

        self.Lci = np.hstack((Lc_m3_i, Lc_m2_i, Lc_m1_i, Lc_d0_i, Lc_p1_i, Lc_p2_i, Lc_p3_i))
        self.Lcj = np.hstack((Lc_m3_j, Lc_m2_j, Lc_m1_j, Lc_d0_j, Lc_p1_j, Lc_p2_j, Lc_p3_j))

    def _get_Lp_indices(self, nx):
        """
        Build the i and j indices for the Lp sparse matrix construction. The Lp matrix for each
        cell has the sparsity pattern of:

        0   X   0   X
        X   0   0   X
        0   X   0   X
        0   0   X   0
                        0   X   0   X
                        X   0   0   X
                        0   X   0   X
                        0   0   X   0
                                        .
                                            .
                                                .
                                                    .
                                                        0   X   0   X
                                                        X   0   0   X
                                                        0   X   0   X
                                                        0   0   X   0
        """

        Lp_m1_i = self.get_indices(nx, [1, 2, 3])
        Lp_m1_j = self.get_indices(nx, [0, 1, 2])
        Lp_p1_i = self.get_indices(nx, [0, 2])
        Lp_p1_j = self.get_indices(nx, [1, 3])
        Lp_p2_i = self.get_indices(nx, [1])
        Lp_p2_j = self.get_indices(nx, [3])
        Lp_p3_i = self.get_indices(nx, [0])
        Lp_p3_j = self.get_indices(nx, [3])

        self.Lpi = np.hstack((Lp_m1_i, Lp_p1_i, Lp_p2_i, Lp_p3_i))
        self.Lpj = np.hstack((Lp_m1_j, Lp_p1_j, Lp_p2_j, Lp_p3_j))

    def _get_eigenvalue_indices(self, nx):
        L_d0_i = self.get_indices(nx, [0, 1, 2, 3])
        L_d0_j = self.get_indices(nx, [0, 1, 2, 3])

        self.Li = L_d0_i
        self.Lj = L_d0_j

    @staticmethod
    def get_indices(size: int, num: list):

        return np.concatenate([np.arange(n, 4 * (size + 1), 4, dtype=np.int) for n in num])
