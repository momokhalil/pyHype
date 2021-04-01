import numpy as np

class XDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, input_, nx):
        self.input = input_

        self.A_d = None
        self.A_m1 = None
        self.A_m2 = None
        self.A_m3 = None
        self.A_p1 = None
        self.A_p2 = None

        self.X_d = None
        self.X_m1 = None
        self.X_m2 = None
        self.X_m3 = None
        self.X_p1 = None
        self.X_p2 = None

        self.Xi_d = None
        self.Xi_m1 = None
        self.Xi_m2 = None
        self.Xi_m3 = None
        self.Xi_p1 = None
        self.Xi_p2 = None
        self.Xi_p3 = None

        self.lam = None

        self.get(nx)

    def get(self, nx):
        self.A_d = np.ones((3 * (nx + 1), 1))
        self.A_m1 = np.ones((3 * (nx + 1), 1))
        self.A_m2 = np.ones((2 * (nx + 1), 1))
        self.A_m3 = np.ones((1 * (nx + 1), 1))
        self.A_p1 = np.ones((2 * (nx + 1), 1))
        self.A_p2 = (self.input.gamma - 1) * np.ones((1 * (nx + 1), 1))

        self.X_d = np.ones((4 * (nx + 1), 1))
        self.X_m1 = np.ones((3 * (nx + 1), 1))
        self.X_m2 = np.ones((2 * (nx + 1), 1))
        self.X_m3 = np.ones((1 * (nx + 1), 1))
        self.X_p1 = np.ones((3 * (nx + 1), 1))
        self.X_p1[2::3] = -1
        self.X_p2 = np.ones((1 * (nx + 1), 1))

        self.Xi_d = np.ones((3 * (nx + 1), 1))
        self.Xi_m1 = -np.ones((3 * (nx + 1), 1))
        self.Xi_m2 = np.ones((1 * (nx + 1), 1))
        self.Xi_m3 = np.ones((1 * (nx + 1), 1))
        self.Xi_p1 = np.ones((3 * (nx + 1), 1))
        self.Xi_p2 = np.ones((2 * (nx + 1), 1))
        self.Xi_p3 = np.ones((1 * (nx + 1), 1))

        self.lam = np.zeros((4 * (nx + 1), 1))

class XDIR_EIGENSYSTEM_INDICES:
    def __init__(self, nx):
        self.Ai = None
        self.Aj = None
        self.Xi = None
        self.Xj = None
        self.Xi_i = None
        self.Xi_j = None
        self.Li = None
        self.Lj = None

        self.get(nx)

    def get(self, nx):
        A_d_i = np.arange(0, 4 * (nx + 1), 1)
        A_d_i = np.delete(A_d_i, slice(0, None, 4))
        A_d_j = A_d_i
        A_d_m1_i = A_d_i
        A_d_m1_j = A_d_j - 1
        A_d_m2_i = A_d_i
        A_d_m2_i = np.delete(A_d_m2_i, slice(0, None, 3))
        A_d_m2_j = A_d_m2_i - 2
        A_d_m3_i = np.arange(3, 4 * (nx + 1), 4)
        A_d_m3_j = A_d_m3_i - 3
        A_d_p1_i = A_d_m2_j
        A_d_p1_j = A_d_m2_i - 1
        A_d_p2_i = np.arange(1, 4 * (nx + 1), 4)
        A_d_p2_j = np.arange(3, 4 * (nx + 1), 4)

        X_d_i = np.arange(0, 4 * (nx + 1), 1)
        X_d_j = X_d_i
        X_d_m1_i = A_d_m1_i
        X_d_m1_j = A_d_m1_j
        X_d_m2_i = A_d_m2_i
        X_d_m2_j = A_d_m2_j
        X_d_m3_i = A_d_m3_i
        X_d_m3_j = A_d_m3_j
        X_d_p1_i = X_d_i
        X_d_p1_i = np.delete(X_d_p1_i, slice(3, None, 4))
        X_d_p1_j = X_d_p1_i + 1
        X_d_p2_i = np.arange(0, 4 * (nx + 1), 4)
        X_d_p2_j = np.arange(2, 4 * (nx + 1), 4)

        Xi_d_i = np.arange(0, 4 * (nx + 1), 1)
        Xi_d_i = np.delete(Xi_d_i, slice(3, None, 4))
        Xi_d_j = Xi_d_i
        Xi_d_m1_i = A_d_m1_i
        Xi_d_m1_j = A_d_m1_j
        Xi_d_m2_i = X_d_p2_j
        Xi_d_m2_j = X_d_p2_i
        Xi_d_m3_i = A_d_m3_i
        Xi_d_m3_j = A_d_m3_j
        Xi_d_p1_i = X_d_p1_i
        Xi_d_p1_j = X_d_p1_j
        Xi_d_p2_i = A_d_m2_j
        Xi_d_p2_j = A_d_m2_i
        Xi_d_p3_i = A_d_m3_j
        Xi_d_p3_j = A_d_m3_i

        L_d_i = np.arange(0, 4 * (nx + 1), 1)
        L_d_j = np.arange(0, 4 * (nx + 1), 1)

        self.Ai = np.hstack((A_d_m3_i, A_d_m2_i, A_d_m1_i, A_d_i, A_d_p1_i, A_d_p2_i))
        self.Aj = np.hstack((A_d_m3_j, A_d_m2_j, A_d_m1_j, A_d_j, A_d_p1_j, A_d_p2_j))
        self.Xi = np.hstack((X_d_m3_i, X_d_m2_i, X_d_m1_i, X_d_i, X_d_p1_i, X_d_p2_i))
        self.Xj = np.hstack((X_d_m3_j, X_d_m2_j, X_d_m1_j, X_d_j, X_d_p1_j, X_d_p2_j))
        self.Xi_i = np.hstack((Xi_d_m3_i, Xi_d_m2_i, Xi_d_m1_i, Xi_d_i, Xi_d_p1_i, Xi_d_p2_i, Xi_d_p3_i))
        self.Xi_j = np.hstack((Xi_d_m3_j, Xi_d_m2_j, Xi_d_m1_j, Xi_d_j, Xi_d_p1_j, Xi_d_p2_j, Xi_d_p3_j))
        self.Li = L_d_i
        self.Lj = L_d_j


class YDIR_EIGENSYSTEM_VECTORS:
    def __init__(self, input_, nx):
        self.input = input_

        self.B_d = None
        self.B_m1 = None
        self.B_m2 = None
        self.B_m3 = None
        self.B_p1 = None
        self.B_p2 = None

        self.X_d = None
        self.X_m1 = None
        self.X_m2 = None
        self.X_m3 = None
        self.X_p1 = None
        self.X_p2 = None

        self.Xi_d = None
        self.Xi_m1 = None
        self.Xi_m2 = None
        self.Xi_m3 = None
        self.Xi_p1 = None
        self.Xi_p2 = None
        self.Xi_p3 = None

        self.lam = None

        self.get(nx)

    def get(self, nx):
        self.B_d = np.ones((3 * (nx + 1), 1))
        self.B_m1 = np.ones((3 * (nx + 1), 1))
        self.B_m2 = np.ones((2 * (nx + 1), 1))
        self.B_m3 = np.ones((1 * (nx + 1), 1))
        self.B_p1 = np.ones((2 * (nx + 1), 1))
        self.B_p1[1::2] = (self.input.gamma - 1)
        self.B_p2 = np.ones((1 * (nx + 1), 1))

        self.X_d = np.ones((4 * (nx + 1), 1))
        self.X_m1 = np.ones((3 * (nx + 1), 1))
        self.X_m2 = np.ones((2 * (nx + 1), 1))
        self.X_m3 = np.ones((1 * (nx + 1), 1))
        self.X_p1 = np.ones((2 * (nx + 1), 1))
        self.X_p2 = np.ones((2 * (nx + 1), 1))

        self.Xi_d = np.ones((3 * (nx + 1), 1))
        self.Xi_m1 = np.ones((2 * (nx + 1), 1))
        self.Xi_m2 = np.ones((2 * (nx + 1), 1))
        self.Xi_m3 = np.ones((1 * (nx + 1), 1))
        self.Xi_p1 = np.ones((3 * (nx + 1), 1))
        self.Xi_p2 = np.ones((2 * (nx + 1), 1))
        self.Xi_p3 = np.ones((1 * (nx + 1), 1))

        self.lam = np.zeros((4 * (nx + 1), 1))


class YDIR_EIGENSYSTEM_INDICES:
    def __init__(self, nx):
        self.Bi = None
        self.Bj = None
        self.Xi = None
        self.Xj = None
        self.Xi_i = None
        self.Xi_j = None
        self.Li = None
        self.Lj = None

        self.get(nx)

    def get(self, nx):
        B_d_i = np.arange(0, 4 * (nx + 1), 1)
        B_d_i = np.delete(B_d_i, slice(None, None, 4))
        B_d_j = B_d_i
        B_d_m1_i = B_d_i
        B_d_m1_j = B_d_j - 1
        B_d_m2_i = B_d_i
        B_d_m2_i = np.delete(B_d_m2_i, slice(None, None, 3))
        B_d_m2_j = B_d_m2_i - 2
        B_d_m3_i = np.arange(3, 4 * (nx + 1), 4)
        B_d_m3_j = B_d_m3_i - 3
        B_d_p1_i = np.arange(0, 4 * (nx + 1), 1)
        B_d_p1_i = np.delete(B_d_p1_i, slice(3, None, 4))
        B_d_p1_i = np.delete(B_d_p1_i, slice(None, None, 3))
        B_d_p1_j = B_d_p1_i + 1
        B_d_p2_i = np.arange(0, 4 * (nx + 1), 4)
        B_d_p2_j = np.arange(2, 4 * (nx + 1), 4)

        X_d_i = np.arange(0, 4 * (nx + 1), 1)
        X_d_j = X_d_i
        X_d_m1_i = B_d_m1_i
        X_d_m1_j = B_d_m1_j
        X_d_m2_i = B_d_m2_i
        X_d_m2_j = B_d_m2_j
        X_d_m3_i = B_d_m3_i
        X_d_m3_j = B_d_m3_j
        X_d_p1_i = X_d_i
        X_d_p1_i = np.delete(X_d_p1_i, slice(2, None, 4))
        X_d_p1_i = np.delete(X_d_p1_i, slice(2, None, 3))
        X_d_p1_j = X_d_p1_i + 1
        X_d_p2_i = B_d_m2_j
        X_d_p2_j = B_d_m2_i

        Xi_d_i = X_d_i
        Xi_d_i = np.delete(Xi_d_i, slice(3, None, 4))
        Xi_d_j = Xi_d_i
        Xi_d_m1_i = X_d_p1_j
        Xi_d_m1_j = X_d_p1_i
        Xi_d_m2_i = X_d_m2_i
        Xi_d_m2_j = X_d_m2_j
        Xi_d_m3_i = X_d_m3_i
        Xi_d_m3_j = X_d_m3_j
        Xi_d_p1_i = X_d_m1_j
        Xi_d_p1_j = X_d_m1_i
        Xi_d_p2_i = Xi_d_m2_j
        Xi_d_p2_j = Xi_d_m2_i
        Xi_d_p3_i = Xi_d_m3_j
        Xi_d_p3_j = Xi_d_m3_i

        L_d_i = np.arange(0, 4 * (nx + 1), 1)
        L_d_j = np.arange(0, 4 * (nx + 1), 1)

        self.Bi = np.hstack((B_d_m3_i, B_d_m2_i, B_d_m1_i, B_d_i, B_d_p1_i, B_d_p2_i))
        self.Bj = np.hstack((B_d_m3_j, B_d_m2_j, B_d_m1_j, B_d_j, B_d_p1_j, B_d_p2_j))
        self.Xi = np.hstack((X_d_m3_i, X_d_m2_i, X_d_m1_i, X_d_i, X_d_p1_i, X_d_p2_i))
        self.Xj = np.hstack((X_d_m3_j, X_d_m2_j, X_d_m1_j, X_d_j, X_d_p1_j, X_d_p2_j))
        self.Xi_i = np.hstack((Xi_d_m3_i, Xi_d_m2_i, Xi_d_m1_i, Xi_d_i, Xi_d_p1_i, Xi_d_p2_i, Xi_d_p3_i))
        self.Xi_j = np.hstack((Xi_d_m3_j, Xi_d_m2_j, Xi_d_m1_j, Xi_d_j, Xi_d_p1_j, Xi_d_p2_j, Xi_d_p3_j))
        self.Li = L_d_i
        self.Lj = L_d_j
