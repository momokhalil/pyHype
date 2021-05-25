import numba
import numpy as np
import scipy.sparse as scisparse
from .states import PrimitiveState, RoePrimitiveState

def xdir_wavespeeds(WL: PrimitiveState, WR: PrimitiveState):
    return WR.u - WR.a(), WR.u + WR.a(), WL.u - WL.a(), WL.u + WL.a()

def harten_correction_xdir(Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
    lambda_R_1, lambda_R_3, lambda_L_1, lambda_L_3 = xdir_wavespeeds(WL, WR)

    theta_1 = 2 * (lambda_R_1 - lambda_L_1)
    theta_3 = 2 * (lambda_R_3 - lambda_L_3)

    theta_1[theta_1 < 0] = 0
    theta_3[theta_3 < 0] = 0

    L1 = Wroe.u - Wroe.a()
    L3 = Wroe.u + Wroe.a()

    idx1 = np.absolute(L1) < theta_1
    idx3 = np.absolute(L3) < theta_3

    L1[idx1] = 0.5 * (np.square(L1[idx1]) / theta_1[idx1] + theta_1[idx1])
    L3[idx3] = 0.5 * (np.square(L3[idx3]) / theta_3[idx3] + theta_3[idx3])

    return L1, L3

def ydir_wavespeeds(WL: PrimitiveState, WR: PrimitiveState):
    return WR.v - WR.a(), WR.v + WR.a(), WL.v - WL.a(), WL.v + WL.a()

def harten_correction_ydir(Wroe: RoePrimitiveState, WL: PrimitiveState, WR: PrimitiveState):
    lambda_R_1, lambda_R_3, lambda_L_1, lambda_L_3 = ydir_wavespeeds(WL, WR)

    theta_1 = 2 * (lambda_R_1 - lambda_L_1)
    theta_3 = 2 * (lambda_R_3 - lambda_L_3)

    theta_1[theta_1 < 0] = 0
    theta_3[theta_3 < 0] = 0

    L1 = Wroe.v - Wroe.a()
    L3 = Wroe.v + Wroe.a()

    idx1 = np.absolute(L1) < theta_1
    idx3 = np.absolute(L3) < theta_3

    L1[idx1] = 0.5 * (np.square(L1[idx1]) / theta_1[idx1] + theta_1[idx1])
    L3[idx3] = 0.5 * (np.square(L3[idx3]) / theta_3[idx3] + theta_3[idx3])

    return L1, L3
