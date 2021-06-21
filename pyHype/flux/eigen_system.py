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
        self.Rc_p1[2::3] = -1
        self.Rc_p2 = np.ones((1 * (nx + 1)))

        self.Lc_d0 = np.ones((3 * (nx + 1)))
        self.Lc_m1 = -np.ones((3 * (nx + 1)))
        self.Lc_m2 = np.ones((1 * (nx + 1)))
        self.Lc_m3 = np.ones((1 * (nx + 1)))
        self.Lc_p1 = np.ones((3 * (nx + 1)))
        self.Lc_p2 = np.ones((2 * (nx + 1)))
        self.Lc_p3 = np.ones((1 * (nx + 1)))

        self.lam = np.zeros((4 * (nx + 1)))


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

        Rc_d0_i = np.arange(0, 4 * (nx + 1), 1)
        Rc_d0_j = Rc_d0_i
        Rc_d0_m1_i = A_d0_m1_i
        Rc_d0_m1_j = A_d0_m1_j
        Rc_d0_m2_i = A_d0_m2_i
        Rc_d0_m2_j = A_d0_m2_j
        Rc_d0_m3_i = A_d0_m3_i
        Rc_d0_m3_j = A_d0_m3_j
        Rc_d0_p1_i = Rc_d0_i
        Rc_d0_p1_i = np.delete(Rc_d0_p1_i, slice(3, None, 4))
        Rc_d0_p1_j = Rc_d0_p1_i + 1
        Rc_d0_p2_i = np.arange(0, 4 * (nx + 1), 4)
        Rc_d0_p2_j = np.arange(2, 4 * (nx + 1), 4)

        Lc_d0_i = np.arange(0, 4 * (nx + 1), 1)
        Lc_d0_i = np.delete(Lc_d0_i, slice(3, None, 4))
        Lc_d0_j = Lc_d0_i
        Lc_d0_m1_i = A_d0_m1_i
        Lc_d0_m1_j = A_d0_m1_j
        Lc_d0_m2_i = Rc_d0_p2_j
        Lc_d0_m2_j = Rc_d0_p2_i
        Lc_d0_m3_i = A_d0_m3_i
        Lc_d0_m3_j = A_d0_m3_j
        Lc_d0_p1_i = Rc_d0_p1_i
        Lc_d0_p1_j = Rc_d0_p1_j
        Lc_d0_p2_i = A_d0_m2_j
        Lc_d0_p2_j = A_d0_m2_i
        Lc_d0_p3_i = A_d0_m3_j
        Lc_d0_p3_j = A_d0_m3_i

        L_d_i = np.arange(0, 4 * (nx + 1), 1)
        L_d_j = np.arange(0, 4 * (nx + 1), 1)

        self.Ai = np.hstack((A_d0_m3_i, A_d0_m2_i, A_d0_m1_i, A_d0_i, A_d0_p1_i, A_d0_p2_i))
        self.Aj = np.hstack((A_d0_m3_j, A_d0_m2_j, A_d0_m1_j, A_d0_j, A_d0_p1_j, A_d0_p2_j))
        self.Lc = np.hstack((Rc_d0_m3_i, Rc_d0_m2_i, Rc_d0_m1_i, Rc_d0_i, Rc_d0_p1_i, Rc_d0_p2_i))
        self.Rcj = np.hstack((Rc_d0_m3_j, Rc_d0_m2_j, Rc_d0_m1_j, Rc_d0_j, Rc_d0_p1_j, Rc_d0_p2_j))
        self.Lc_i = np.hstack((Lc_d0_m3_i, Lc_d0_m2_i, Lc_d0_m1_i, Lc_d0_i, Lc_d0_p1_i, Lc_d0_p2_i, Lc_d0_p3_i))
        self.Lc_j = np.hstack((Lc_d0_m3_j, Lc_d0_m2_j, Lc_d0_m1_j, Lc_d0_j, Lc_d0_p1_j, Lc_d0_p2_j, Lc_d0_p3_j))
        self.Li = L_d_i
        self.Lj = L_d_j
