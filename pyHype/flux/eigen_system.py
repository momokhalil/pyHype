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

import numpy as np


class XDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, inputs, nx):

        self.inputs = inputs

        self.A_d0 = np.ones((1, 3 * (nx + 1)))
        self.A_m1 = np.ones((1, 3 * (nx + 1)))
        self.A_m2 = np.ones((1, 2 * (nx + 1)))
        self.A_m3 = np.ones((1, 1 * (nx + 1)))
        self.A_p1 = np.ones((1, 2 * (nx + 1)))
        self.A_p2 = np.full((1, 1 * (nx + 1)), self.inputs.gamma - 1)

        self.X_d0 = np.ones((1, 4 * (nx + 1)))
        self.X_m1 = np.ones((1, 3 * (nx + 1)))
        self.X_m2 = np.ones((1, 2 * (nx + 1)))
        self.X_m3 = np.ones((1, 1 * (nx + 1)))
        self.X_p1 = np.ones((1, 3 * (nx + 1)))
        self.X_p1[0, 2::3] = -1
        self.X_p2 = np.ones((1, 1 * (nx + 1)))

        self.Xi_d0 = np.ones((1, 3 * (nx + 1)))
        self.Xi_m1 = -np.ones((1, 3 * (nx + 1)))
        self.Xi_m2 = np.ones((1, 1 * (nx + 1)))
        self.Xi_m3 = np.ones((1, 1 * (nx + 1)))
        self.Xi_p1 = np.ones((1, 3 * (nx + 1)))
        self.Xi_p2 = np.ones((1, 2 * (nx + 1)))
        self.Xi_p3 = np.ones((1, 1 * (nx + 1)))

        self.lam = np.zeros((1, 4 * (nx + 1)))


class XDIR_EIGENSYSTEM_INDICES:
    def __init__(self, nx):

        A_d0_i = np.arange(0, 4 * (nx + 1), 1)
        A_d0_i = np.delete(A_d0_i, slice(0, None, 4))
        A_d0_j = A_d0_i
        A_d0_m1_i = A_d0_i
        A_d0_m1_j = A_d0_j - 1
        A_d0_m2_i = A_d0_i
        A_d0_m2_i = np.delete(A_d0_m2_i, slice(0, None, 3))
        A_d0_m2_j = A_d0_m2_i - 2
        A_d0_m3_i = np.arange(3, 4 * (nx + 1), 4)
        A_d0_m3_j = A_d0_m3_i - 3
        A_d0_p1_i = A_d0_m2_j
        A_d0_p1_j = A_d0_m2_i - 1
        A_d0_p2_i = np.arange(1, 4 * (nx + 1), 4)
        A_d0_p2_j = np.arange(3, 4 * (nx + 1), 4)

        X_d0_i = np.arange(0, 4 * (nx + 1), 1)
        X_d0_j = X_d0_i
        X_d0_m1_i = A_d0_m1_i
        X_d0_m1_j = A_d0_m1_j
        X_d0_m2_i = A_d0_m2_i
        X_d0_m2_j = A_d0_m2_j
        X_d0_m3_i = A_d0_m3_i
        X_d0_m3_j = A_d0_m3_j
        X_d0_p1_i = X_d0_i
        X_d0_p1_i = np.delete(X_d0_p1_i, slice(3, None, 4))
        X_d0_p1_j = X_d0_p1_i + 1
        X_d0_p2_i = np.arange(0, 4 * (nx + 1), 4)
        X_d0_p2_j = np.arange(2, 4 * (nx + 1), 4)

        Xi_d0_i = np.arange(0, 4 * (nx + 1), 1)
        Xi_d0_i = np.delete(Xi_d0_i, slice(3, None, 4))
        Xi_d0_j = Xi_d0_i
        Xi_d0_m1_i = A_d0_m1_i
        Xi_d0_m1_j = A_d0_m1_j
        Xi_d0_m2_i = X_d0_p2_j
        Xi_d0_m2_j = X_d0_p2_i
        Xi_d0_m3_i = A_d0_m3_i
        Xi_d0_m3_j = A_d0_m3_j
        Xi_d0_p1_i = X_d0_p1_i
        Xi_d0_p1_j = X_d0_p1_j
        Xi_d0_p2_i = A_d0_m2_j
        Xi_d0_p2_j = A_d0_m2_i
        Xi_d0_p3_i = A_d0_m3_j
        Xi_d0_p3_j = A_d0_m3_i

        L_d_i = np.arange(0, 4 * (nx + 1), 1)
        L_d_j = np.arange(0, 4 * (nx + 1), 1)

        self.Ai = np.hstack((A_d0_m3_i, A_d0_m2_i, A_d0_m1_i, A_d0_i, A_d0_p1_i, A_d0_p2_i))
        self.Aj = np.hstack((A_d0_m3_j, A_d0_m2_j, A_d0_m1_j, A_d0_j, A_d0_p1_j, A_d0_p2_j))
        self.Xi = np.hstack((X_d0_m3_i, X_d0_m2_i, X_d0_m1_i, X_d0_i, X_d0_p1_i, X_d0_p2_i))
        self.Xj = np.hstack((X_d0_m3_j, X_d0_m2_j, X_d0_m1_j, X_d0_j, X_d0_p1_j, X_d0_p2_j))
        self.Xi_i = np.hstack((Xi_d0_m3_i, Xi_d0_m2_i, Xi_d0_m1_i, Xi_d0_i, Xi_d0_p1_i, Xi_d0_p2_i, Xi_d0_p3_i))
        self.Xi_j = np.hstack((Xi_d0_m3_j, Xi_d0_m2_j, Xi_d0_m1_j, Xi_d0_j, Xi_d0_p1_j, Xi_d0_p2_j, Xi_d0_p3_j))
        self.Li = L_d_i
        self.Lj = L_d_j


class YDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, inputs, nx):

        self.inputs = inputs

        self.B_d0 = np.ones((1, 3 * (nx + 1)))
        self.B_m1 = np.ones((1, 3 * (nx + 1)))
        self.B_m2 = np.ones((1, 2 * (nx + 1)))
        self.B_m3 = np.ones((1, 1 * (nx + 1)))
        self.B_p1 = np.ones((1, 2 * (nx + 1)))
        self.B_p1[0, 1::2] = (self.inputs.gamma - 1)
        self.B_p2 = np.ones((1, 1 * (nx + 1)))

        self.X_d0 = np.ones((1, 4 * (nx + 1)))
        self.X_m1 = np.ones((1, 3 * (nx + 1)))
        self.X_m2 = np.ones((1, 2 * (nx + 1)))
        self.X_m3 = np.ones((1, 1 * (nx + 1)))
        self.X_p1 = np.ones((1, 2 * (nx + 1)))
        self.X_p2 = np.ones((1, 2 * (nx + 1)))

        self.Xi_d0 = np.ones((1, 3 * (nx + 1)))
        self.Xi_m1 = np.ones((1, 2 * (nx + 1)))
        self.Xi_m2 = np.ones((1, 2 * (nx + 1)))
        self.Xi_m3 = np.ones((1, 1 * (nx + 1)))
        self.Xi_p1 = np.ones((1, 3 * (nx + 1)))
        self.Xi_p2 = np.ones((1, 2 * (nx + 1)))
        self.Xi_p3 = np.ones((1, 1 * (nx + 1)))

        self.lam = np.zeros((1, 4 * (nx + 1)))


class YDIR_EIGENSYSTEM_INDICES:
    def __init__(self, nx):

        B_d0_i = np.arange(0, 4 * (nx + 1), 1)
        B_d0_i = np.delete(B_d0_i, slice(None, None, 4))
        B_d0_j = B_d0_i
        B_d0_m1_i = B_d0_i
        B_d0_m1_j = B_d0_j - 1
        B_d0_m2_i = B_d0_i
        B_d0_m2_i = np.delete(B_d0_m2_i, slice(None, None, 3))
        B_d0_m2_j = B_d0_m2_i - 2
        B_d0_m3_i = np.arange(3, 4 * (nx + 1), 4)
        B_d0_m3_j = B_d0_m3_i - 3
        B_d0_p1_i = np.arange(0, 4 * (nx + 1), 1)
        B_d0_p1_i = np.delete(B_d0_p1_i, slice(3, None, 4))
        B_d0_p1_i = np.delete(B_d0_p1_i, slice(None, None, 3))
        B_d0_p1_j = B_d0_p1_i + 1
        B_d0_p2_i = np.arange(0, 4 * (nx + 1), 4)
        B_d0_p2_j = np.arange(2, 4 * (nx + 1), 4)

        X_d0_i = np.arange(0, 4 * (nx + 1), 1)
        X_d0_j = X_d0_i
        X_d0_m1_i = B_d0_m1_i
        X_d0_m1_j = B_d0_m1_j
        X_d0_m2_i = B_d0_m2_i
        X_d0_m2_j = B_d0_m2_j
        X_d0_m3_i = B_d0_m3_i
        X_d0_m3_j = B_d0_m3_j
        X_d0_p1_i = X_d0_i
        X_d0_p1_i = np.delete(X_d0_p1_i, slice(2, None, 4))
        X_d0_p1_i = np.delete(X_d0_p1_i, slice(2, None, 3))
        X_d0_p1_j = X_d0_p1_i + 1
        X_d0_p2_i = B_d0_m2_j
        X_d0_p2_j = B_d0_m2_i

        Xi_d0_i = X_d0_i
        Xi_d0_i = np.delete(Xi_d0_i, slice(3, None, 4))
        Xi_d0_j = Xi_d0_i
        Xi_d0_m1_i = X_d0_p1_j
        Xi_d0_m1_j = X_d0_p1_i
        Xi_d0_m2_i = X_d0_m2_i
        Xi_d0_m2_j = X_d0_m2_j
        Xi_d0_m3_i = X_d0_m3_i
        Xi_d0_m3_j = X_d0_m3_j
        Xi_d0_p1_i = X_d0_m1_j
        Xi_d0_p1_j = X_d0_m1_i
        Xi_d0_p2_i = Xi_d0_m2_j
        Xi_d0_p2_j = Xi_d0_m2_i
        Xi_d0_p3_i = Xi_d0_m3_j
        Xi_d0_p3_j = Xi_d0_m3_i

        L_d_i = np.arange(0, 4 * (nx + 1), 1)
        L_d_j = np.arange(0, 4 * (nx + 1), 1)

        self.Bi = np.hstack((B_d0_m3_i, B_d0_m2_i, B_d0_m1_i, B_d0_i, B_d0_p1_i, B_d0_p2_i))
        self.Bj = np.hstack((B_d0_m3_j, B_d0_m2_j, B_d0_m1_j, B_d0_j, B_d0_p1_j, B_d0_p2_j))
        self.Xi = np.hstack((X_d0_m3_i, X_d0_m2_i, X_d0_m1_i, X_d0_i, X_d0_p1_i, X_d0_p2_i))
        self.Xj = np.hstack((X_d0_m3_j, X_d0_m2_j, X_d0_m1_j, X_d0_j, X_d0_p1_j, X_d0_p2_j))
        self.Xi_i = np.hstack((Xi_d0_m3_i, Xi_d0_m2_i, Xi_d0_m1_i, Xi_d0_i, Xi_d0_p1_i, Xi_d0_p2_i, Xi_d0_p3_i))
        self.Xi_j = np.hstack((Xi_d0_m3_j, Xi_d0_m2_j, Xi_d0_m1_j, Xi_d0_j, Xi_d0_p1_j, Xi_d0_p2_j, Xi_d0_p3_j))
        self.Li = L_d_i
        self.Lj = L_d_j
