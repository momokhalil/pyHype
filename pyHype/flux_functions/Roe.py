import numba
from numba import float32
import numpy as np
import scipy.sparse as sparse
from pyHype.flux_functions.base import FluxFunction
from pyHype.states import PrimitiveState, RoePrimitiveState, ConservativeState
from pyHype.utils import harten_correction_xdir, harten_correction_ydir
from pyHype.flux_functions.eigen_system import XDIR_EIGENSYSTEM_INDICES, \
                                               XDIR_EIGENSYSTEM_VECTORS, \
                                               YDIR_EIGENSYSTEM_INDICES, \
                                               YDIR_EIGENSYSTEM_VECTORS

class ROE_FLUX_X(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = XDIR_EIGENSYSTEM_VECTORS(self.inputs, self.nx)

        self.A_d0, self.A_m1, self.A_m2, self.A_m3, self.A_p1, self.A_p2 \
            = vec.A_d0, vec.A_m1, vec.A_m2, vec.A_m3, vec.A_p1, vec.A_p2

        self.X_d0, self.X_m1, self.X_m2, self.X_m3, self.X_p1, self.X_p2 \
            = vec.X_d0, vec.X_m1, vec.X_m2, vec.X_m3, vec.X_p1, vec.X_p2

        self.Xi_d0, self.Xi_m1, self.Xi_m2, self.Xi_m3, self.Xi_p1, self.Xi_p2, self.Xi_p3 \
            = vec.Xi_d0, vec.Xi_m1, vec.Xi_m2, vec.Xi_m3, vec.Xi_p1, vec.Xi_p2, vec.Xi_p3

        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = XDIR_EIGENSYSTEM_INDICES(self.nx)

        self.Ai, self.Aj = idx.Ai, idx.Aj
        self.Xi, self.Xj = idx.Xi, idx.Xj
        self.Xi_i, self.Xi_j = idx.Xi_i, idx.Xi_j
        self.Li, self.Lj = idx.Li, idx.Lj

        # Build sparse matrices
        data = stack(self.A_m3, self.A_m2, self.A_m1, self.A_d0, self.A_p1, self.A_p2)
        self.A = sparse.coo_matrix((data, (self.Ai, self.Aj)))

        data = stack(self.X_m3, self.X_m2, self.X_m1, self.X_d0, self.X_p1, self.X_p2)
        self.X = sparse.coo_matrix((data, (self.Xi, self.Xj)))

        data = stack(self.Xi_m3, self.Xi_m2, self.Xi_m1, self.Xi_d0, self.Xi_p1, self.Xi_p2, self.Xi_p3)
        self.Xi = sparse.coo_matrix((data, (self.Xi_i, self.Xi_j)))

        data = np.zeros((4 * self.nx + 4))
        self.Lambda = sparse.coo_matrix((data, (self.Li, self.Lj)))

    def _get_eigen_system_from_roe_state(self) -> None:

        # Create Left and Right PrimitiveStates
        WL, WR = self._L.to_W(), self._L.to_W()

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, self.nx + 1, WL=WL, WR=WR)

        # Harten entropy correction
        Lm, Lp = harten_correction_xdir(Wroe, WL, WR)

        # Calculate quantities to construct eigensystem
        a       = Wroe.a()
        gu      = self.g * Wroe.u
        gbu     = self.gb * Wroe.u
        gtu     = self.gt * Wroe.u
        ghu     = self.gh * Wroe.u
        ghv     = self.gh * Wroe.v
        gtv     = self.gt * Wroe.v
        u2      = Wroe.u ** 2
        v2      = Wroe.v ** 2
        uv      = Wroe.u * Wroe.v
        ek      = 0.5 * (u2 + v2)
        a2      = a ** 2
        ta2     = a2 * 2
        ua      = Wroe.u * a
        ghek    = self.gh * ek
        H       = Wroe.H()

        self.A_m3           = Wroe.u * (ghek - H)
        self.A_m2[0::2]     = -uv
        self.A_m2[1::2]     = H - self.gh * u2
        self.A_m1[0::3]     = ghek - u2
        self.A_m1[1::3]     = Wroe.v
        self.A_m1[2::3]     = -self.gh * uv
        self.A_d0[0::3]     = gbu
        self.A_d0[1::3]     = Wroe.u
        self.A_d0[2::3]     = gu
        self.A_p1[1::2]     = -ghv

        self.A.data = stack(self.A_m3, self.A_m2, self.A_m1, self.A_d0, self.A_p1, self.A_p2)

        self.X_m3           = H - ua
        self.X_m2[0::2]     = Wroe.v
        self.X_m2[1::2]     = ek
        self.X_m1[0::3]     = Lm
        self.X_m1[1::3]     = Wroe.v
        self.X_m1[2::3]     = H + ua
        self.X_d0[1::4]     = Wroe.u
        self.X_d0[2::4]     = Wroe.v
        self.X_d0[3::4]     = Wroe.v
        self.X_p1[1::3]     = Lp

        self.X.data = stack(self.X_m3, self.X_m2, self.X_m1, self.X_d0, self.X_p1, self.X_p2)

        self.Xi_m3          = Wroe.v
        self.Xi_m2          = (ghek - ua) / ta2
        self.Xi_m1[0::3]    = (a2 - ghek) / a2
        self.Xi_m1[1::3]    = (gtu + a) / ta2
        self.Xi_d0[0::3]    = (ghek + ua) / ta2
        self.Xi_d0[1::3]    = ghu / a2
        self.Xi_d0[2::3]    = gtv / ta2
        self.Xi_p1[0::3]    = (gtu - a) / ta2
        self.Xi_p1[1::3]    = ghv / a2
        self.Xi_p1[2::3]    = self.gh / ta2
        self.Xi_p2[0::2]    = gtv / ta2
        self.Xi_p2[1::2]    = self.gt / a2
        self.Xi_p3          = self.gh / ta2

        self.Xi.data = stack(self.Xi_m3, self.Xi_m2, self.Xi_m1, self.Xi_d0, self.Xi_p1, self.Xi_p2, self.Xi_p3)

        self.lam[0::4]      = Lm
        self.lam[1::4]      = Wroe.u
        self.lam[2::4]      = Lp
        self.lam[3::4]      = Wroe.u

        self.Lambda.data = self.lam.reshape(-1, )


    def get_flux(self):
        self._get_eigen_system_from_roe_state()
        return 0.5 * (self.A.dot(self.L_plus_R()) + self.X.dot(np.absolute(self.Lambda).dot(self.Xi.dot(self.dULR()))))


class ROE_FLUX_Y(FluxFunction):
    def __init__(self, inputs):
        super().__init__(inputs)

        # Thermodynamic quantities
        self.gh = self.g - 1
        self.gt = 1 - self.g
        self.gb = 3 - self.g

        # X-direction eigensystem data vectors
        vec = YDIR_EIGENSYSTEM_VECTORS(self.inputs, self.ny)

        self.B_d0, self.B_m1, self.B_m2, self.B_m3, self.B_p1, self.B_p2 \
            = vec.B_d0, vec.B_m1, vec.B_m2, vec.B_m3, vec.B_p1, vec.B_p2

        self.X_d0, self.X_m1, self.X_m2, self.X_m3, self.X_p1, self.X_p2 \
            = vec.X_d0, vec.X_m1, vec.X_m2, vec.X_m3, vec.X_p1, vec.X_p2

        self.Xi_d0, self.Xi_m1, self.Xi_m2, self.Xi_m3, self.Xi_p1, self.Xi_p2, self.Xi_p3 \
            = vec.Xi_d0, vec.Xi_m1, vec.Xi_m2, vec.Xi_m3, vec.Xi_p1, vec.Xi_p2, vec.Xi_p3

        self.lam = vec.lam

        # X-direction eigensystem indices
        idx = YDIR_EIGENSYSTEM_INDICES(self.ny)

        self.Bi, self.Bj = idx.Bi, idx.Bj
        self.Xi, self.Xj = idx.Xi, idx.Xj
        self.Xi_i, self.Xi_j = idx.Xi_i, idx.Xi_j
        self.Li, self.Lj = idx.Li, idx.Lj

        # Build sparse matrices
        data = stack(self.B_m3, self.B_m2, self.B_m1, self.B_d0, self.B_p1, self.B_p2)
        self.B = sparse.coo_matrix((data, (self.Bi, self.Bj)))

        data = stack(self.X_m3, self.X_m2, self.X_m1, self.X_d0, self.X_p1, self.X_p2)
        self.X = sparse.coo_matrix((data, (self.Xi, self.Xj)))

        data = stack(self.Xi_m3, self.Xi_m2, self.Xi_m1, self.Xi_d0, self.Xi_p1, self.Xi_p2, self.Xi_p3)
        self.Xi = sparse.coo_matrix((data, (self.Xi_i, self.Xi_j)))

        data = np.zeros((4 * self.nx + 4))
        self.Lambda = sparse.coo_matrix((data, (self.Li, self.Lj)))


    def _get_eigen_system_from_roe_state(self) -> None:

        # Create Left and Right PrimitiveStates
        WL, WR = self._L.to_W(), self._L.to_W()

        # Get Roe state
        Wroe = RoePrimitiveState(self.inputs, self.nx + 1, WL=WL, WR=WR)

        # Harten entropy correction
        Lm, Lp = harten_correction_ydir(Wroe, WL, WR)

        # Calculate quantities to construct eigensystem
        a       = Wroe.a()
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
        a2      = a ** 2
        ta2     = a2 * 2
        va      = Wroe.v * a
        ghek    = self.gh * ek
        H       = Wroe.H()

        self.B_m3           = Wroe.v * (ghek - H)
        self.B_m2[0::2]     = ghek - v2
        self.B_m2[1::2]     = -self.gh * uv
        self.B_m1[0::3]     = -uv
        self.B_m1[1::3]     = -ghu
        self.B_m1[2::3]     = H - self.gh * v2
        self.B_d0[0::3]     = Wroe.v
        self.B_d0[1::3]     = gbv
        self.B_d0[2::3]     = gv
        self.B_p1[0::2]     = Wroe.u

        self.B.data = stack(self.B_m3, self.B_m2, self.B_m1, self.B_d0, self.B_p1, self.B_p2)

        self.X_m3           = H - va
        self.X_m2[0::2]     = Lm
        self.X_m2[1::2]     = ek
        self.X_m1[0::3]     = Wroe.u
        self.X_m1[1::3]     = Wroe.v
        self.X_m1[2::3]     = H + va
        self.X_d0[1::4]     = Wroe.u
        self.X_d0[2::4]     = Lp
        self.X_d0[3::4]     = Wroe.u
        self.X_p1[1::2]     = Wroe.u

        self.X.data = stack(self.X_m3, self.X_m2, self.X_m1, self.X_d0, self.X_p1, self.X_p2)

        self.Xi_m3          = Wroe.u
        self.Xi_m2[0::2]    = (ghek - va) / ta2
        self.Xi_m1[0::2]    = (a2 - ghek) / a2
        self.Xi_m1[1::2]    = gtu / ta2
        self.Xi_d0[0::3]    = (ghek + va) / ta2
        self.Xi_d0[1::3]    = ghu / a2
        self.Xi_d0[2::3]    = (gtv + a) / ta2
        self.Xi_p1[0::3]    = gtu / ta2
        self.Xi_p1[1::3]    = ghv / a2
        self.Xi_p1[2::3]    = self.gh / ta2
        self.Xi_p2[0::2]    = (gtv - a) / ta2
        self.Xi_p2[1::2]    = self.gt / a2
        self.Xi_p3          = self.gh / ta2

        self.Xi.data = stack(self.Xi_m3, self.Xi_m2, self.Xi_m1, self.Xi_d0, self.Xi_p1, self.Xi_p2, self.Xi_p3)

        self.lam[0::4]      = Lm
        self.lam[1::4]      = Wroe.v
        self.lam[2::4]      = Lp
        self.lam[3::4]      = Wroe.v

        self.Lambda.data    = self.lam.reshape(-1, )

    def get_flux(self):
        self._get_eigen_system_from_roe_state()
        return 0.5 * (self.B.dot(self.L_plus_R()) + self.X.dot(np.absolute(self.Lambda).dot(self.Xi.dot(self.dULR()))))

def stack(*args):
    return np.row_stack(args).reshape(-1, )
