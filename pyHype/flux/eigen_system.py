import numpy as np
from numba.experimental import jitclass
from pyHype.flux import numba_spec as ns


@jitclass(ns.XDIR_EIGENSYSTEM_VECTORS_SPEC)
class XDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, inputs, nx):

        self.inputs = inputs

        self.A_d0 = np.ones((3 * (nx + 1), 1))
        self.A_m1 = np.ones((3 * (nx + 1), 1))
        self.A_m2 = np.ones((2 * (nx + 1), 1))
        self.A_m3 = np.ones((1 * (nx + 1), 1))
        self.A_p1 = np.ones((2 * (nx + 1), 1))
        self.A_p2 = np.full((1 * (nx + 1), 1), self.inputs.gamma - 1)

        self.X_d0 = np.ones((4 * (nx + 1), 1))
        self.X_m1 = np.ones((3 * (nx + 1), 1))
        self.X_m2 = np.ones((2 * (nx + 1), 1))
        self.X_m3 = np.ones((1 * (nx + 1), 1))
        self.X_p1 = np.ones((3 * (nx + 1), 1))
        self.X_p1[2::3] = -1
        self.X_p2 = np.ones((1 * (nx + 1), 1))

        self.Xi_d0 = np.ones((3 * (nx + 1), 1))
        self.Xi_m1 = -np.ones((3 * (nx + 1), 1))
        self.Xi_m2 = np.ones((1 * (nx + 1), 1))
        self.Xi_m3 = np.ones((1 * (nx + 1), 1))
        self.Xi_p1 = np.ones((3 * (nx + 1), 1))
        self.Xi_p2 = np.ones((2 * (nx + 1), 1))
        self.Xi_p3 = np.ones((1 * (nx + 1), 1))

        self.lam = np.zeros((4 * (nx + 1), 1))


@jitclass(ns.XDIR_EIGENSYSTEM_INDICES_SPEC)
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


@jitclass(ns.YDIR_EIGENSYSTEM_VECTORS_SPEC)
class YDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, inputs, nx):

        self.inputs = inputs

        self.B_d0 = np.ones((3 * (nx + 1), 1))
        self.B_m1 = np.ones((3 * (nx + 1), 1))
        self.B_m2 = np.ones((2 * (nx + 1), 1))
        self.B_m3 = np.ones((1 * (nx + 1), 1))
        self.B_p1 = np.ones((2 * (nx + 1), 1))
        self.B_p1[1::2] = (self.inputs.gamma - 1)
        self.B_p2 = np.ones((1 * (nx + 1), 1))

        self.X_d0 = np.ones((4 * (nx + 1), 1))
        self.X_m1 = np.ones((3 * (nx + 1), 1))
        self.X_m2 = np.ones((2 * (nx + 1), 1))
        self.X_m3 = np.ones((1 * (nx + 1), 1))
        self.X_p1 = np.ones((2 * (nx + 1), 1))
        self.X_p2 = np.ones((2 * (nx + 1), 1))

        self.Xi_d0 = np.ones((3 * (nx + 1), 1))
        self.Xi_m1 = np.ones((2 * (nx + 1), 1))
        self.Xi_m2 = np.ones((2 * (nx + 1), 1))
        self.Xi_m3 = np.ones((1 * (nx + 1), 1))
        self.Xi_p1 = np.ones((3 * (nx + 1), 1))
        self.Xi_p2 = np.ones((2 * (nx + 1), 1))
        self.Xi_p3 = np.ones((1 * (nx + 1), 1))

        self.lam = np.zeros((4 * (nx + 1), 1))


@jitclass(ns.YDIR_EIGENSYSTEM_INDICES_SPEC)
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
