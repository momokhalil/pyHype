import time
import numpy as np
from numba import jit as jit
from numba.experimental import jitclass
from pyHype.input.implosion import implosion
from pyHype.states.states import PrimitiveState
import pyHype.mesh.mesh_builder as mesh_builder
from pyHype.states.states import primitivestate_type
import pyHype.input.input_file_builder as input_file_builder


class test_outer_jit:
    def __init__(self, W1_, W2_):
        self.W1 = W1_
        self.W2 = W2_

    def roe(self):
        get_roe_state_vec_jit(self.W1.rho, self.W1.u, self.W1.v, self.W1.H(),
                              self.W2.rho, self.W2.u, self.W2.v, self.W2.H())

@jitclass([('W1', primitivestate_type),
           ('W2', primitivestate_type)])
class test_inner_jit:
    def __init__(self, W1_, W2_):
        self.W1 = W1_
        self.W2 = W2_

    def roe(self):
        """
        Compute *Roe* average quantities as described earlier

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        # Compute common quantities
        sqRhoL = np.sqrt(self.W1.rho)
        sqRhoR = np.sqrt(self.W2.rho)
        sqRhoRL = sqRhoL + sqRhoR

        # Compute *Roe* average quantities
        rho = np.sqrt(self.W1.rho * self.W2.rho)
        u = (self.W1.u * sqRhoL + self.W2.u * sqRhoR) / sqRhoRL
        v = (self.W1.v * sqRhoL + self.W2.v * sqRhoR) / sqRhoRL
        H = (self.W1.H() * sqRhoL + self.W2.H() * sqRhoR) / sqRhoRL
        p = (1.4 - 1) / 1.4 * rho * (H - 0.5 * (u ** 2 + v ** 2))


class test_no_jit:
    def __init__(self, W1_, W2_):
        self.W1 = W1_
        self.W2 = W2_

    def roe(self):
        """
        Compute *Roe* average quantities as described earlier

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        # Compute common quantities
        sqRhoL = np.sqrt(self.W1.rho)
        sqRhoR = np.sqrt(self.W2.rho)
        sqRhoRL = sqRhoL + sqRhoR

        # Compute *Roe* average quantities
        rho = np.sqrt(self.W1.rho * self.W2.rho)
        u = (self.W1.u * sqRhoL + self.W2.u * sqRhoR) / sqRhoRL
        v = (self.W1.v * sqRhoL + self.W2.v * sqRhoR) / sqRhoRL
        H = (self.W1.H() * sqRhoL + self.W2.H() * sqRhoR) / sqRhoRL
        p = (1.4 - 1) / 1.4 * rho * (H - 0.5 * (u ** 2 + v ** 2))


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


@jit(nopython=True)
def get_roe_state_vec_jit(rhoL, uL, vL, HL, rhoR, uR, vR, HR):
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

    num = 2000

    _W1 = np.random.random((4 * num, 1))
    _W2 = np.random.random((4 * num, 1))

    inputsdict = implosion
    mesh_inputs = mesh_builder.build(mesh_name=inputsdict['mesh_name'],
                                             nx=inputsdict['nx'],
                                             ny=inputsdict['ny'])

    inputs = input_file_builder.build(inputsdict, mesh_inputs)

    W1 = PrimitiveState(inputs, num)
    W1.from_state_vector(_W1)

    W2 = PrimitiveState(inputs, num)
    W2.from_state_vector(_W2)

    get_roe_state(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    get_roe_state_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())



    itr = 1000

    """t1 = time.time()
    for i in range(itr):
        get_roe_state(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t1_ = time.time()

    get_roe_state_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t2 = time.time()
    for i in range(itr):
        get_roe_state_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t2_ = time.time()

    t3 = time.time()
    for i in range(itr):
        get_roe_state_vec(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t3_ = time.time()

    get_roe_state_vec_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t4 = time.time()
    for i in range(itr):
        get_roe_state_vec_jit(W1.rho, W1.u, W1.v, W1.H(), W2.rho, W2.u, W2.v, W2.H())
    t4_ = time.time()

    print(t1_ - t1)
    print(t2_ - t2)
    print(t3_ - t3)
    print(t4_ - t4)"""

    out = test_outer_jit(W1, W2)
    inn = test_inner_jit(W1, W2)
    no = test_no_jit(W1, W2)

    out.roe()
    inn.roe()
    no.roe()

    t_out = 0
    for i in range(itr):
        t1 = time.time()
        out.roe()
        t2 = time.time()
        t_out += t2 - t1

    t_in = 0
    for i in range(itr):
        t1 = time.time()
        inn.roe()
        t2 = time.time()
        t_in += t2 - t1

    t_no = 0
    for i in range(itr):
        t1 = time.time()
        no.roe()
        t2 = time.time()
        t_no += t2 - t1

    print(t_no, t_in, t_out)
