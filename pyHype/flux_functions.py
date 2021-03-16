import numpy as np
from scipy.sparse import bsr_matrix as bsrmat
from abc import ABC, abstractmethod
from .states import RoePrimitiveState, PrimitiveState, ConservativeState
from .utils import *

class FluxFunction(ABC):
    def __init__(self, input_):
        self._input = input_
        self._L = None
        self._R = None
        self.g = input_.get('gamma')

    def set_left_state(self, U_L):
        self._L = U_L

    def set_right_state(self, U_R):
        self._R = U_R

    def L_minus_R(self):
        return self._L - self._R

    def L_plus_R(self):
        return self._L + self._R

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
        self.A_p2 = (self.input.get('gamma') - 1) * np.ones((1 * (nx + 1), 1))

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
    def __init__(self, input_, nx):
        self._input = input_
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
        A_diag_i        = np.arange(0, 4*(nx + 1), 1)
        np.delete(A_diag_i, slice(None, None, 4))
        A_diag_j        = A_diag_i
        A_diag_m1_i     = A_diag_i
        A_diag_m1_j     = A_diag_j - 1
        A_diag_m2_i     = A_diag_i
        np.delete(A_diag_m2_i, slice(0, None, 3))
        A_diag_m2_j     = A_diag_m2_i - 2
        A_diag_m3_i     = np.arange(3, 4*(nx + 1), 4)
        A_diag_m3_j     = A_diag_m3_i - 3
        A_diag_p1_i     = A_diag_m2_j
        A_diag_p1_j     = A_diag_m2_i - 1
        A_diag_p2_i     = np.arange(1, 4*(nx + 1), 4)
        A_diag_p2_j     = np.arange(3, 4*(nx + 1), 4)

        X_diag_i        = np.arange(0, 4*(nx + 1), 1)
        X_diag_j        = X_diag_i
        X_diag_m1_i     = A_diag_m1_i
        X_diag_m1_j     = A_diag_m1_j
        X_diag_m2_i     = A_diag_m2_i
        X_diag_m2_j     = A_diag_m2_j
        X_diag_m3_i     = A_diag_m3_i
        X_diag_m3_j     = A_diag_m3_j
        X_diag_p1_i     = X_diag_i
        np.delete(X_diag_p1_i, slice(3, None, 4))
        X_diag_p1_j     = X_diag_p1_i + 1
        X_diag_p2_i     = np.arange(0, 4*(nx + 1), 4)
        X_diag_p2_j     = np.arange(2, 4*(nx + 1), 4)

        Xi_diag_i       = np.arange(0, 4*(nx + 1), 1)
        np.delete(Xi_diag_i, slice(3, None, 4))
        Xi_diag_j       = Xi_diag_i
        Xi_diag_m1_i    = A_diag_m1_i
        Xi_diag_m1_j    = A_diag_m1_j
        Xi_diag_m2_i    = X_diag_p2_j
        Xi_diag_m2_j    = X_diag_p2_i
        Xi_diag_m3_i    = A_diag_m3_i
        Xi_diag_m3_j    = A_diag_m3_j
        Xi_diag_p1_i    = X_diag_p1_i
        Xi_diag_p1_j    = X_diag_p1_j
        Xi_diag_p2_i    = A_diag_m2_j
        Xi_diag_p2_j    = A_diag_m2_i
        Xi_diag_p3_i    = A_diag_m3_j
        Xi_diag_p3_j    = A_diag_m3_i

        L_diag_i        = np.arange(0, 4 * (nx + 1), 1)
        L_diag_j        = np.arange(0, 4 * (nx + 1), 1)

        self.Ai         = np.vstack((A_diag_m3_i, A_diag_m2_i, A_diag_m1_i, A_diag_i, A_diag_p1_i, A_diag_p2_i))
        self.Aj         = np.vstack((A_diag_m3_j, A_diag_m2_j, A_diag_m1_j, A_diag_j, A_diag_p1_j, A_diag_p2_j))
        self.Xi         = np.vstack((X_diag_m3_i, X_diag_m2_i, X_diag_m1_i, X_diag_i, X_diag_p1_i, X_diag_p2_i))
        self.Xj         = np.vstack((X_diag_m3_j, X_diag_m2_j, X_diag_m1_j, X_diag_j, X_diag_p1_j, X_diag_p2_j))
        self.Xi_i       = np.vstack((Xi_diag_m3_i, Xi_diag_m2_i, Xi_diag_m1_i, Xi_diag_i, Xi_diag_p1_i, Xi_diag_p2_i, Xi_diag_p3_i))
        self.Xi_j       = np.vstack((Xi_diag_m3_j, Xi_diag_m2_j, Xi_diag_m1_j, Xi_diag_j, Xi_diag_p1_j, Xi_diag_p2_j, Xi_diag_p3_j))
        self.Li         = L_diag_i
        self.Lj         = L_diag_j


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
        self.B_p1[1::2] = (self.input.get('gamma') - 1)
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
    def __init__(self, input_, nx):
        self._input = input_
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
        B_diag_i = np.arange(0, 4 * (nx + 1), 1)
        np.delete(B_diag_i, slice(None, None, 4))
        B_diag_j = B_diag_i
        B_diag_m1_i = B_diag_i
        B_diag_m1_j = B_diag_j - 1
        B_diag_m2_i = B_diag_i
        np.delete(B_diag_m2_i, slice(None, None, 3))
        B_diag_m2_j = B_diag_m2_i - 2
        B_diag_m3_i = np.arange(3, 4 * (nx + 1), 4)
        B_diag_m3_j = B_diag_m3_i - 3
        B_diag_p1_i = np.arange(0, 4 * (nx + 1), 1)
        np.delete(B_diag_p1_i, slice(3, None, 4))
        np.delete(B_diag_p1_i, slice(None, None, 3))
        B_diag_p1_j = B_diag_p1_i + 1
        B_diag_p2_i = np.arange(0, 4 * (nx + 1), 4)
        B_diag_p2_j = np.arange(2, 4 * (nx + 1), 4)

        X_diag_i = np.arange(0, 4 * (nx + 1), 1)
        X_diag_j = X_diag_i
        X_diag_m1_i = B_diag_m1_i
        X_diag_m1_j = B_diag_m1_j
        X_diag_m2_i = B_diag_m2_i
        X_diag_m2_j = B_diag_m2_j
        X_diag_m3_i = B_diag_m3_i
        X_diag_m3_j = B_diag_m3_j
        X_diag_p1_i = X_diag_i
        np.delete(X_diag_p1_i, slice(2, None, 4))
        np.delete(X_diag_p1_i, slice(2, None, 3))
        X_diag_p1_j = X_diag_p1_i + 1
        X_diag_p2_i = B_diag_m2_j
        X_diag_p2_j = B_diag_m2_i

        Xi_diag_i = X_diag_i
        np.delete(Xi_diag_i, slice(3, None, 4))
        Xi_diag_j = Xi_diag_i
        Xi_diag_m1_i = X_diag_p1_j
        Xi_diag_m1_j = X_diag_p1_i
        Xi_diag_m2_i = X_diag_m2_i
        Xi_diag_m2_j = X_diag_m2_j
        Xi_diag_m3_i = X_diag_m3_i
        Xi_diag_m3_j = X_diag_m3_j
        Xi_diag_p1_i = X_diag_m1_j
        Xi_diag_p1_j = X_diag_m1_i
        Xi_diag_p2_i = Xi_diag_m2_j
        Xi_diag_p2_j = Xi_diag_m2_i
        Xi_diag_p3_i = Xi_diag_m3_j
        Xi_diag_p3_j = Xi_diag_m3_i

        L_diag_i = np.arange(0, 4 * (nx + 1), 1)
        L_diag_j = np.arange(0, 4 * (nx + 1), 1)

        self.Bi = np.vstack((B_diag_m3_i, B_diag_m2_i, B_diag_m1_i, B_diag_i, B_diag_p1_i, B_diag_p2_i))
        self.Bj = np.vstack((B_diag_m3_j, B_diag_m2_j, B_diag_m1_j, B_diag_j, B_diag_p1_j, B_diag_p2_j))
        self.Xi = np.vstack((X_diag_m3_i, X_diag_m2_i, X_diag_m1_i, X_diag_i, X_diag_p1_i, X_diag_p2_i))
        self.Xj = np.vstack((X_diag_m3_j, X_diag_m2_j, X_diag_m1_j, X_diag_j, X_diag_p1_j, X_diag_p2_j))
        self.Xi_i = np.vstack((Xi_diag_m3_i, Xi_diag_m2_i, Xi_diag_m1_i, Xi_diag_i, Xi_diag_p1_i, Xi_diag_p2_i, Xi_diag_p3_i))
        self.Xi_j = np.vstack((Xi_diag_m3_j, Xi_diag_m2_j, Xi_diag_m1_j, Xi_diag_j, Xi_diag_p1_j, Xi_diag_p2_j, Xi_diag_p3_j))
        self.Li = L_diag_i
        self.Lj = L_diag_j


class ROE_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g
        self.idx = XDIR_EIGENSYSTEM_INDICES(self._input, input_.get('mesh_inputs').get('nx'))
        self.vec = XDIR_EIGENSYSTEM_VECTORS(self._input, input_.get('mesh_inputs').get('nx'))

    def _get_eigen_system_from_roe_state(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):

        Lm, Lp  = harten_correction_xdir(Wroe, WL, WR)
        gu      = self.g * Wroe.u
        gbu     = self.gb * Wroe.u
        gtu     = self.gt * Wroe.u
        ghu     = self.gh * Wroe.u
        ghv     = self.gh * Wroe.v
        gtv     = self.gt * Wroe.v
        u2      = Wroe.u ^ 2
        v2      = Wroe.v ^ 2
        uv      = Wroe.u * Wroe.v
        ek      = 0.5 * (u2 + v2)
        a2      = Wroe.a() ^ 2
        ta2     = a2 * 2
        ua      = Wroe.u * Wroe.a
        ghek    = self.gh * ek
        H       = Wroe.H()

        self.vec.A_m3           = Wroe.u * (ghek - H)
        self.vec.A_m2[::2]      = -uv
        self.vec.A_m2[1::2]     = H - self.gh * u2
        self.vec.A_m1[::3]      = ghek - u2
        self.vec.A_m1[1::3]     = Wroe.v
        self.vec.A_m1[2::3]     = -self.gh * uv
        self.vec.A_d[::3]       = gbu
        self.vec.A_d[1::3]      = Wroe.u
        self.vec.A_d[2::3]      = gu
        self.vec.A_p1[1::2]     = -ghv
        A = bsrmat((np.vstack((self.vec.A_m3, self.vec.A_m2, self.vec.A_m1,
                               self.vec.A_d,
                               self.vec.A_p1, self.vec.A_p2)),
                   (self.idx.Ai, self.idx.Aj)))

        self.vec.X_m3           = H - ua
        self.vec.X_m2[::2]      = Wroe.v
        self.vec.X_m2[1::2]     = ek
        self.vec.X_m1[::3]      = Lm
        self.vec.X_m1[1::3]     = Wroe.v
        self.vec.X_m1[2::3]     = H + ua
        self.vec.X_d[1::4]      = Wroe.u
        self.vec.X_d[2::4]      = Wroe.v
        self.vec.X_d[3::4]      = Wroe.v
        self.vec.X_p1[1::3]     = Lp
        X = bsrmat((np.vstack((self.vec.X_m3, self.vec.X_m2, self.vec.X_m1,
                               self.vec.X_d,
                               self.vec.X_p1, self.vec.X_p2)),
                   (self.idx.Xi, self.idx.Xj)))

        self.vec.Xi_m3          = Wroe.v
        self.vec.Xi_m2          = (ghek - ua) / ta2
        self.vec.Xi_m1[::3]     = (a2 - ghek) / a2
        self.vec.Xi_m1[1::3]    = (gtu + Wroe.a()) / ta2
        self.vec.Xi_d[::3]      = (ghek + ua) / ta2
        self.vec.Xi_d[1::3]     = ghu / a2
        self.vec.Xi_d[2::3]     = gtv / ta2
        self.vec.Xi_p1[::3]     = (gtu - Wroe.a()) / ta2
        self.vec.Xi_p1[1::3]    = ghv / a2
        self.vec.Xi_p1[2::3]    = self.gh / ta2
        self.vec.Xi_p2[::2]     = gtv / ta2
        self.vec.Xi_p2[1::2]    = self.gt / a2
        self.vec.Xi_p3          = self.gh / ta2
        Xi = bsrmat(np.vstack((self.vec.Xi_m3, self.vec.Xi_m2, self.vec.Xi_m1,
                               self.vec.Xi_d,
                               self.vec.Xi_p1, self.vec.Xi_p2, self.vec.Xi_p3)),
                    (self.idx.Xi_i, self.idx.Xi_j))

        self.vec.lam[0::4]      = Lm
        self.vec.lam[1::4]      = Wroe.u
        self.vec.lam[2::4]      = Lp
        self.vec.lam[3::4]      = Wroe.u
        Lambda = bsrmat((self.vec.lam, (self.idx.Li, self.idx.Lj)))

        return A, X, Xi, Lambda

    def get_flux(self):
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.get('mesh_inputs').get('n'), WL=WL, WR=WR)
        A, X, Xi, Lambda = self._get_eigen_system_from_roe_state(Wroe, WL, WR)
        return 0.5 * (A.multiply(self.L_plus_R()) + X.multiply(Lambda.multiply(Xi.multiply(self.L_minus_R()))))


class ROE_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g
        self.idx = YDIR_EIGENSYSTEM_INDICES(self._input, input_.get('mesh_inputs').get('ny'))
        self.vec = YDIR_EIGENSYSTEM_VECTORS(self._input, input_.get('mesh_inputs').get('ny'))

    def _get_eigen_system_from_roe_state(self, Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):

        Lm, Lp  = harten_correction_ydir(Wroe, WL, WR)
        gv      = self.g * Wroe.v
        gbv     = self.gb * Wroe.v
        gtu     = self.gt * Wroe.u
        ghu     = self.gh * Wroe.u
        ghv     = self.gh * Wroe.v
        gtv     = self.gt * Wroe.v
        u2      = Wroe.u ** 2
        v2      = Wroe.v ** 2
        uv      = Wroe.u * Wroe.v
        ek      = 0.5 * (u2 + v2)
        a2      = Wroe.a() ** 2
        ta2     = a2 * 2
        va      = Wroe.v * Wroe.a
        ghek    = self.gh * ek
        H       = Wroe.H()

        self.vec.B_m3           = Wroe.v * (ghek - H)
        self.vec.B_m2[::2]      = ghek - v2
        self.vec.B_m2[1::2]     = -self.gh * uv
        self.vec.B_m1[::3]      = -uv
        self.vec.B_m1[1::3]     = -ghu
        self.vec.B_m1[2::3]     = H - self.gh * v2
        self.vec.B_d[::3]       = Wroe.v
        self.vec.B_d[1::3]      = gbv
        self.vec.B_d[2::3]      = gv
        self.vec.B_p1[0::2]     = Wroe.u
        B = bsrmat((np.vstack((self.vec.B_m3, self.vec.B_m2, self.vec.B_m1,
                               self.vec.B_d,
                               self.vec.B_p1, self.vec.B_p2)),
                   (self.idx.Bi, self.idx.Bj)))

        self.vec.X_m3           = H - va
        self.vec.X_m2[::2]      = Lm
        self.vec.X_m2[1::2]     = ek
        self.vec.X_m1[::3]      = Wroe.u
        self.vec.X_m1[1::3]     = Wroe.v
        self.vec.X_m1[2::3]     = H + va
        self.vec.X_d[1::4]      = Wroe.u
        self.vec.X_d[2::4]      = Lp
        self.vec.X_d[3::4]      = Wroe.u
        self.vec.X_p1[1::2]     = Wroe.u
        X = bsrmat((np.vstack((self.vec.X_m3, self.vec.X_m2, self.vec.X_m1,
                               self.vec.X_d,
                               self.vec.X_p1, self.vec.X_p2)),
                   (self.idx.Xi, self.idx.Xj)))

        self.vec.Xi_m3          = Wroe.u
        self.vec.Xi_m2[::2]     = (ghek - va) / ta2
        self.vec.Xi_m1[::2]     = (a2 - ghek) / a2
        self.vec.Xi_m1[1::2]    = gtu / ta2
        self.vec.Xi_d[::3]      = (ghek + va) / ta2
        self.vec.Xi_d[1::3]     = ghu / a2
        self.vec.Xi_d[2::3]     = (gtv + Wroe.a()) / ta2
        self.vec.Xi_p1[::3]     = gtu / ta2
        self.vec.Xi_p1[1::3]    = ghv / a2
        self.vec.Xi_p1[2::3]    = self.gh / ta2
        self.vec.Xi_p2[::2]     = (gtv - Wroe.a()) / ta2
        self.vec.Xi_p2[1::2]    = self.gt / a2
        self.vec.Xi_p3          = self.gh / ta2
        Xi = bsrmat((np.vstack((self.vec.Xi_m3, self.vec.Xi_m2, self.vec.Xi_m1,
                                self.vec.Xi_d,
                                self.vec.Xi_p1, self.vec.Xi_p2, self.vec.Xi_p3)),
                    (self.idx.Xi_i, self.idx.Xi_j)))

        self.vec.lam[0::4]      = Lm
        self.vec.lam[1::4]      = Wroe.v
        self.vec.lam[2::4]      = Lp
        self.vec.lam[3::4]      = Wroe.v
        Lambda = bsrmat((self.vec.lam, (self.idx.Li, self.idx.Lj)))

        return B, X, Xi, Lambda

    def get_flux(self):
        WL, WR = self._L.to_W(), self._L.to_W()
        Wroe = RoePrimitiveState(self._input, self._input.get('mesh_inputs').get('n'), WL=WL, WR=WR)
        B, X, Xi, Lambda = self._get_eigen_system_from_roe_state(Wroe, WL, WR)
        return 0.5 * (B.multiply(self.L_plus_R()) + X.multiply(Lambda.multiply(Xi.multiply(self.L_minus_R()))))


class HLLE_FLUX_X(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass


class HLLE_FLUX_Y(FluxFunction):
    def __init__(self, input_):
        super().__init__(input_)

    def get_flux(self):
        pass


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
