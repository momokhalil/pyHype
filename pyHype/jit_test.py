import numba as nb
from numba import jit as jit
import numpy as np
from pyHype.states.states import PrimitiveState
import pyHype.mesh.mesh_builder as mesh_builder
from pyHype.input.implosion import implosion
import pyHype.input.input_file_builder as input_file_builder
import time


@jit(nopython=True)
def get_roe_state_jit(rhoL, uL, vL, HL, rhoR, uR, vR, HR):
    """
    Compute *Roe* average quantities as described earlier

    **Parameters**                                              \n
        WL      Left primitive state                            \n
        WR      Right primitive state                           \n
    """

    # Compute common quantities
    sqRhoL  = np.sqrt(rhoL)
    sqRhoR  = np.sqrt(rhoR)
    sqRhoRL = sqRhoL + sqRhoR

    s = rhoL.shape[0]

    rho = np.empty((s, 1))
    u = np.empty((s, 1))
    v = np.empty((s, 1))
    p = np.empty((s, 1))
    H = np.empty((s, 1))

    # Compute *Roe* average quantities
    for i in range(s):

        rho[i]     = np.sqrt(rhoL[i] * rhoR[i])
        u[i]       = (uL[i] * sqRhoL[i] + uR[i] * sqRhoR[i]) / sqRhoRL[i]
        v[i]       = (vL[i] * sqRhoL[i] + vR[i] * sqRhoR[i]) / sqRhoRL[i]
        H[i]       = (HL[i] * sqRhoL[i] + HR[i] * sqRhoR[i]) / sqRhoRL[i]
        p[i]       = (1.4 - 1) / 1.4 * rho[i] * (H[i] - 0.5 * (u[i]**2 + v[i]**2))


def get_roe_state(rhoL, uL, vL, HL, rhoR, uR, vR, HR):
    """
    Compute *Roe* average quantities as described earlier

    **Parameters**                                              \n
        WL      Left primitive state                            \n
        WR      Right primitive state                           \n
    """

    # Compute common quantities
    sqRhoL = np.sqrt(rhoL)
    sqRhoR = np.sqrt(rhoR)
    sqRhoRL = sqRhoL + sqRhoR

    # Compute *Roe* average quantities
    s = rhoL.shape[0]

    rho = np.empty((s, 1))
    u = np.empty((s, 1))
    v = np.empty((s, 1))
    p = np.empty((s, 1))
    H = np.empty((s, 1))

    # Compute *Roe* average quantities
    for i in range(s):
        rho[i] = np.sqrt(rhoL[i] * rhoR[i])
        u[i] = (uL[i] * sqRhoL[i] + uR[i] * sqRhoR[i]) / sqRhoRL[i]
        v[i] = (vL[i] * sqRhoL[i] + vR[i] * sqRhoR[i]) / sqRhoRL[i]
        H[i] = (HL[i] * sqRhoL[i] + HR[i] * sqRhoR[i]) / sqRhoRL[i]
        p[i] = (1.4 - 1) / 1.4 * rho[i] * (H[i] - 0.5 * (u[i] ** 2 + v[i] ** 2))


def get_roe_state_vec(rhoL, uL, vL, HL, rhoR, uR, vR, HR):
    """
    Compute *Roe* average quantities as described earlier

    **Parameters**                                              \n
        WL      Left primitive state                            \n
        WR      Right primitive state                           \n
    """

    # Compute common quantities
    sqRhoL = np.sqrt(rhoL)
    sqRhoR = np.sqrt(rhoR)
    sqRhoRL = sqRhoL + sqRhoR

    # Compute *Roe* average quantities
    rho = np.sqrt(rhoL * rhoR)
    u = (uL * sqRhoL + uR * sqRhoR) / sqRhoRL
    v = (vL * sqRhoL + vR * sqRhoR) / sqRhoRL
    H = (HL * sqRhoL + HR * sqRhoR) / sqRhoRL
    p = (1.4 - 1) / 1.4 * rho * (H - 0.5 * (u ** 2 + v ** 2))


if __name__ == '__main__':

    num = 200

    _W1 = np.random.random((4 * num, 1))
    _W2 = np.random.random((4 * num, 1))

    inputsdict = implosion
    meshinputss = mesh_builder.build(mesh_name=inputsdict['mesh_name'],
                                             nx=inputsdict['nx'],
                                             ny=inputsdict['ny'])

    inputs = input_file_builder.build(inputsdict, meshinputss)

    W1 = PrimitiveState(inputs, num)
    W1.from_state_vector(_W1)

    W2 = PrimitiveState(inputs, num)
    W2.from_state_vector(_W2)

    get_roe_state(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    get_roe_state_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())

    t = time.time()

    itr = 1000

    for i in range(itr):
        get_roe_state(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())

    t1 = time.time()

    for i in range(itr):
        get_roe_state_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())

    t2 = time.time()

    for i in range(itr):
        get_roe_state_vec(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())

    t3 = time.time()

    print(t1 - t)
    print(t2 - t1)
    print(t3 - t2)
