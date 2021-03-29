import numba
import numpy as np
from numba import float32
import scipy as sp
import scipy.sparse as sparse
from abc import ABC, abstractmethod
from pyHype.states import PrimitiveState, RoePrimitiveState, ConservativeState
from pyHype.utils import harten_correction_xdir, harten_correction_ydir


class FluxFunction(ABC):
    def __init__(self, input_):
        self._input = input_
        self._L = None
        self._R = None
        self.g = input_.gamma
        self.nx = input_.nx
        self.ny = input_.ny
        self.n = input_.n

    def set_left_state(self, UL):
        self._L = UL

    def set_right_state(self, UR):
        self._R = UR

    def dULR(self):
        return self._L.U - self._R.U

    def L_plus_R(self):
        return self._L.U + self._R.U

    @abstractmethod
    def get_flux(self):
        pass


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


class ROE_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self._input, self.nx)

        self.A_d, self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 \
            = vec.A_d, vec.A_m1, vec.A_m2, vec.A_m3, vec.A_p1, vec.A_p2

        self.X_d, self.X_m1, self.X_m2, self.X_m3, self.X_p1, self.X_p2 \
            = vec.X_d, vec.X_m1, vec.X_m2, vec.X_m3, vec.X_p1, vec.X_p2

        self.Xi_d, self.Xi_m1, self.Xi_m2, self.Xi_m3, self.Xi_p1, self.Xi_p2, self.Xi_p3 \
            = vec.Xi_d, vec.Xi_m1, vec.Xi_m2, vec.Xi_m3, vec.Xi_p1, vec.Xi_p2, vec.Xi_p3

        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(self.nx)

        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Xi, self.Xj = idx.Xi, idx.Xj
        self.Xi_i, self.Xi_j = idx.Xi_i, idx.Xi_j
        self.Li, self.Lj = idx.Li, idx.Lj

    def _get_eig_vars(self, uroe, vroe, aroe, Hroe):

        gu      = self.g * uroe
        gbu     = self.gb * uroe
        gtu     = self.gt * uroe
        ghu     = self.gh * uroe
        ghv     = self.gh * vroe
        gtv     = self.gt * vroe
        u2      = uroe ** 2
        v2      = vroe ** 2
        uv      = uroe * vroe
        ek      = 0.5 * (u2 + v2)
        a2      = aroe ** 2
        ta2     = a2 * 2
        ua      = uroe * aroe
        ghek    = self.gh * ek
        H       = Hroe

        return gu, gbu, gtu, ghu, ghv, gtv, u2, v2, uv, ek, a2, ta2, ua, ghek, H

    def _get_eigen_system_from_roe_state(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):

        Lm, Lp = harten_correction_xdir(Wroe, WL, WR)

        gu, gbu, gtu, ghu, ghv, gtv, u2, v2, uv, ek, a2, ta2, ua, ghek, H = self._get_eig_vars(Wroe.u, Wroe.v,
                                                                                               Wroe.a(), Wroe.H())
        self.A_m3 = Wroe.u * (ghek - H)
        self.A_m2[::2] = -uv
        self.A_m2[1::2] = H - self.gh * u2
        self.A_m1[::3] = ghek - u2
        self.A_m1[1::3] = Wroe.v
        self.A_m1[2::3] = -self.gh * uv
        self.A_d[::3] = gbu
        self.A_d[1::3] = Wroe.u
        self.A_d[2::3] = gu
        self.A_p1[1::2] = -ghv

        data = np.row_stack((self.A_m3,
                             self.A_m2,
                             self.A_m1,
                             self.A_d,
                             self.A_p1,
                             self.A_p2))

        A = sparse.csc_matrix((data.reshape(-1, ), (self.Ai, self.Aj)))

        self.X_m3 = H - ua
        self.X_m2[::2] = Wroe.v
        self.X_m2[1::2] = ek
        self.X_m1[::3] = Lm
        self.X_m1[1::3] = Wroe.v
        self.X_m1[2::3] = H + ua
        self.X_d[1::4] = Wroe.u
        self.X_d[2::4] = Wroe.v
        self.X_d[3::4] = Wroe.v
        self.X_p1[1::3] = Lp

        data = np.row_stack((self.X_m3,
                             self.X_m2,
                             self.X_m1,
                             self.X_d,
                             self.X_p1,
                             self.X_p2))

        X = sparse.csc_matrix((data.reshape(-1, ), (self.Xi, self.Xj)))

        self.Xi_m3          = Wroe.v
        self.Xi_m2          = (ghek - ua) / ta2
        self.Xi_m1[::3]     = (a2 - ghek) / a2
        self.Xi_m1[1::3]    = (gtu + Wroe.a()) / ta2
        self.Xi_d[::3]      = (ghek + ua) / ta2
        self.Xi_d[1::3]     = ghu / a2
        self.Xi_d[2::3]     = gtv / ta2
        self.Xi_p1[::3]     = (gtu - Wroe.a()) / ta2
        self.Xi_p1[1::3]    = ghv / a2
        self.Xi_p1[2::3]    = self.gh / ta2
        self.Xi_p2[::2]     = gtv / ta2
        self.Xi_p2[1::2]    = self.gt / a2
        self.Xi_p3          = self.gh / ta2

        data = np.row_stack((self.Xi_m3,
                             self.Xi_m2,
                             self.Xi_m1,
                             self.Xi_d,
                             self.Xi_p1,
                             self.Xi_p2,
                             self.Xi_p3))

        Xi = sparse.csc_matrix((data.reshape(-1, ), (self.Xi_i, self.Xi_j)))

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.u
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.u

        Lambda = sparse.csc_matrix((self.lam.reshape(-1, ), (self.Li, self.Lj)))

        return A, X, Xi, Lambda

    def get_flux(self):
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.nx + 1, WL=WL, WR=WR)
        A, X, Xi, Lambda = self._get_eigen_system_from_roe_state(Wroe, WL, WR)
        return 0.5 * (A.dot(self.L_plus_R()) + X.dot(np.absolute(Lambda).dot(Xi.dot(self.dULR()))))


class ROE_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = YDIR_EIGENSYSTEM_VECTORS(self._input, self.ny)

        self.B_d, self.B_m1, self.B_m2, self.B_m3, self.B_p1, self.B_p2 \
            = vec.B_d, vec.B_m1, vec.B_m2, vec.B_m3, vec.B_p1, vec.B_p2

        self.X_d, self.X_m1, self.X_m2, self.X_m3, self.X_p1, self.X_p2 \
            = vec.X_d, vec.X_m1, vec.X_m2, vec.X_m3, vec.X_p1, vec.X_p2

        self.Xi_d, self.Xi_m1, self.Xi_m2, self.Xi_m3, self.Xi_p1, self.Xi_p2, self.Xi_p3 \
            = vec.Xi_d, vec.Xi_m1, vec.Xi_m2, vec.Xi_m3, vec.Xi_p1, vec.Xi_p2, vec.Xi_p3

        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = YDIR_EIGENSYSTEM_INDICES(self.ny)
        self.Bi, self.Bj = idx.Bi, idx.Bj
        self.Xi, self.Xj = idx.Xi, idx.Xj
        self.Xi_i, self.Xi_j = idx.Xi_i, idx.Xi_j
        self.Li, self.Lj = idx.Li, idx.Lj


    def _get_eigen_system_from_roe_state(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)
        gv = self.g * Wroe.v
        gbv = self.gb * Wroe.v
        gtu = self.gt * Wroe.u
        ghu = self.gh * Wroe.u
        ghv = self.gh * Wroe.v
        gtv = self.gt * Wroe.v
        u2 = Wroe.u ** 2
        v2 = Wroe.v ** 2
        uv = Wroe.u * Wroe.v
        ek = 0.5 * (u2 + v2)
        a2 = Wroe.a() ** 2
        ta2 = a2 * 2
        va = Wroe.v * Wroe.a()
        ghek = self.gh * ek
        H = Wroe.H()

        self.B_m3 = Wroe.v * (ghek - H)
        self.B_m2[::2] = ghek - v2
        self.B_m2[1::2] = -self.gh * uv
        self.B_m1[::3] = -uv
        self.B_m1[1::3] = -ghu
        self.B_m1[2::3] = H - self.gh * v2
        self.B_d[::3] = Wroe.v
        self.B_d[1::3] = gbv
        self.B_d[2::3] = gv
        self.B_p1[0::2] = Wroe.u

        data = np.row_stack((self.B_m3,
                             self.B_m2,
                             self.B_m1,
                             self.B_d,
                             self.B_p1,
                             self.B_p2))

        B = sparse.csc_matrix((data.reshape(-1, ), (self.Bi, self.Bj)))

        self.X_m3 = H - va
        self.X_m2[::2] = Lm
        self.X_m2[1::2] = ek
        self.X_m1[::3] = Wroe.u
        self.X_m1[1::3] = Wroe.v
        self.X_m1[2::3] = H + va
        self.X_d[1::4] = Wroe.u
        self.X_d[2::4] = Lp
        self.X_d[3::4] = Wroe.u
        self.X_p1[1::2] = Wroe.u

        data = np.row_stack((self.X_m3,
                             self.X_m2,
                             self.X_m1,
                             self.X_d,
                             self.X_p1,
                             self.X_p2))

        X = sparse.csc_matrix((data.reshape(-1, ), (self.Xi, self.Xj)))

        self.Xi_m3 = Wroe.u
        self.Xi_m2[::2] = (ghek - va) / ta2
        self.Xi_m1[::2] = (a2 - ghek) / a2
        self.Xi_m1[1::2] = gtu / ta2
        self.Xi_d[::3] = (ghek + va) / ta2
        self.Xi_d[1::3] = ghu / a2
        self.Xi_d[2::3] = (gtv + Wroe.a()) / ta2
        self.Xi_p1[::3] = gtu / ta2
        self.Xi_p1[1::3] = ghv / a2
        self.Xi_p1[2::3] = self.gh / ta2
        self.Xi_p2[::2] = (gtv - Wroe.a()) / ta2
        self.Xi_p2[1::2] = self.gt / a2
        self.Xi_p3 = self.gh / ta2

        data = np.row_stack((self.Xi_m3,
                             self.Xi_m2,
                             self.Xi_m1,
                             self.Xi_d,
                             self.Xi_p1,
                             self.Xi_p2,
                             self.Xi_p3))

        Xi = sparse.csc_matrix((data.reshape(-1, ), (self.Xi_i, self.Xi_j)))

        self.lam[0::4] = Lm
        self.lam[1::4] = Wroe.v
        self.lam[2::4] = Lp
        self.lam[3::4] = Wroe.v
        Lambda = sparse.csc_matrix((self.lam.reshape(-1, ), (self.Li, self.Lj)))

        return B, X, Xi, Lambda

    def get_flux(self):
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.ny + 1, WL=WL, WR=WR)
        B, X, Xi, Lambda = self._get_eigen_system_from_roe_state(Wroe, WL, WR)
        return 0.5 * (B.dot(self.L_plus_R()) + X.dot(np.absolute(Lambda).dot(Xi.dot(self.dULR()))))


class HLLE_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        flux = np.empty((4 * self._input.nx + 4, 1))
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.nx + 1, WL=WL, WR=WR)

        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)

        lambda_p = np.maximum(WL.u - WL.a(), Lp)
        lambda_m = np.maximum(WR.u - WR.a(), Lm)

        Fl = self._L.get_flux_X()
        Fr = self._R.get_flux_X()

        for i in range(1, self._input.nx + 1):
            l_p = lambda_p[i - 1]
            l_m = lambda_m[i - 1]

            if l_m >= 0:
                flux[4 * i - 4:4 * i] = Fl[4 * i - 4:4 * i]
            elif l_p <= 0:
                flux[4 * i - 4:4 * i] = Fr[4 * i - 4:4 * i]
            else:
                flux[4 * i - 4:4 * i] = l_p * Fl[4 * i - 4:4 * i] - l_m * Fr[4 * i - 4:4 * i] \
                                        + l_p * l_m * (self._R.U[4 * i - 4:4 * i] - self._L.U[4 * i - 4:4 * i]) / (
                                                    l_p - l_m)

        return flux


class HLLE_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        flux = np.empty((4 * self._input.ny + 4, 1))
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.nx + 1, WL=WL, WR=WR)

        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)

        lambda_p = np.maximum(WL.v - WL.a(), Lp)
        lambda_m = np.maximum(WR.v - WR.a(), Lm)

        Gl = self._L.get_flux_Y()
        Gr = self._R.get_flux_Y()

        for i in range(1, self._input.ny + 1):
            l_p = lambda_p[i - 1]
            l_m = lambda_m[i - 1]

            if l_m >= 0:
                flux[4 * i - 4:4 * i] = Gl[4 * i - 4:4 * i]
            elif l_p <= 0:
                flux[4 * i - 4:4 * i] = Gr[4 * i - 4:4 * i]
            else:
                flux[4 * i - 4:4 * i] = l_p * Gl[4 * i - 4:4 * i] - l_m * Gr[4 * i - 4:4 * i] \
                                        + l_p * l_m * (self._R.U[4 * i - 4:4 * i] - self._L.U[4 * i - 4:4 * i]) / (
                                                    l_p - l_m)
        return flux


class HLLL_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass


class HLLL_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass
