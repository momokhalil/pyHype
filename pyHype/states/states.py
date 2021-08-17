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
from __future__ import annotations

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numba as nb
import numpy as np
from pyHype.states.base import State
from pyHype.input.input_file_builder import ProblemInput
import pyHype.states._cstate_jit_funcs as _cjf
from profilehooks import profile
from typing import Callable
import functools


def cache(func: Callable):
    @functools.wraps(func)
    def _wrapper(self, *args):
        if func.__name__ not in self.cache.keys():
            #print('Caching ' + func.__name__ + ' in ' + str(self.__class__))
            self.cache[func.__name__] = func(self, *args)
            return self.cache[func.__name__]
        else:
            #print('Using cached ' + func.__name__ + ' in ' + str(self.__class__))
            return self.cache[func.__name__]
    return _wrapper

class PrimitiveState(State):
    """
    #Primitive Solution State#
    A class that represents the solution state vector of the 2D inviscid Euler equations in primitive form. The
    primitive solution vector, $W$, is represented mathematically as:
    $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$, where $\\rho$ is the solution density,
    $u$ is the velocity in the $x$-direction, $v$ is the velocity in the $y$-direction, and $p$ is the solution
    pressure. `PrimitiveState` can be used to represent the solution state of a QuadBlock in a solver that utilizes
    a primitive formulation. It can also be used to represent the solution state in BoundaryBlocks in a solver that
    utilizes a primitive formulation. Another primary use-case for `PrimitiveState` is converting a `ConservativeState`
    into `PrimitiveState` in order to access primitive solution variables if needed (e.g. flux functions).
    """

    RHO_IDX = 0
    U_IDX = 1
    V_IDX = 2
    P_IDX = 3

    def __init__(self,
                 inputs: ProblemInput,
                 nx: int = None,
                 ny: int = None,
                 state: State = None,
                 U_vector: np.ndarray = None,
                 W_vector: np.ndarray = None):
        """
        ## Attributes

        **Public**                                  \n
            W       Primitive state vector          \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Initialize attributes for storing solution data

        # Check if an input vector is given
        if state or (U_vector is not None) or (W_vector is not None):
            _vec_given = True
        else:
            _vec_given = False

        # Check if more than one input vector is given
        if _vec_given and sum(map(bool, [state, U_vector is not None, W_vector is not None])) > 1:
            raise ValueError('Please provide only one type of input state or vector')

        # Get vector size
        _nx, _ny = 0, 0

        if ny and nx and not _vec_given:
            _nx, _ny = int(nx), int(ny)
        elif _vec_given:
            if state:
                _nx, _ny = state.nx, state.ny
            elif U_vector is not None:
                _nx, _ny = int(U_vector.shape[1]), int(U_vector.shape[0])
            elif W_vector is not None:
                _nx, _ny = int(W_vector.shape[1]), int(W_vector.shape[0])
        else:
            ValueError('Please either provide the size of the vector or a state class or a state vector')

        super().__init__(inputs, _nx, _ny)

        if state:
            if isinstance(state, PrimitiveState):
                self.from_primitive_state(state)
            elif isinstance(state, ConservativeState):
                self.from_conservative_state(state)

        elif U_vector is not None:
            self.from_conservative_state_vector(U_vector)

        elif W_vector is not None:
            self.from_primitive_state_vector(W_vector)

    # ------------------------------------------------------------------------------------------------------------------
    # Define primitive state properties

    # Density property (refers to q0 in Q)
    @property
    def rho(self) -> np.ndarray:
        return self.q0

    @rho.setter
    def rho(self,
            rho: np.ndarray
            ) -> None:
        self.q0 = rho

    # u property (refers to q1 in Q)
    @property
    def u(self) -> np.ndarray:
        return self.q1

    @u.setter
    def u(self,
          u: np.ndarray
          ) -> None:
        self.q1 = u

    # v property (refers to q2 in Q)
    @property
    def v(self) -> np.ndarray:
        return self.q2

    @v.setter
    def v(self,
          v: np.ndarray
          ) -> None:
        self.q2 = v

    # p property (refers to q3 in Q)
    @property
    def p(self) -> np.ndarray:
        return self.q3

    @p.setter
    def p(self,
          p: np.ndarray
          ) -> None:
        self.q3 = p

    # W property (refers to q3 in Q)
    @property
    def W(self) -> np.ndarray:
        return self.Q

    # W property (refers to q3 in Q)
    @W.setter
    def W(self,
          W: np.ndarray
          ) -> None:
        self.Q = W

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS FOR UPDATING INTERNAL STATE BASED ON EXTERNAL INPUTS

    def from_conservative_state(self,
                                U: 'ConservativeState'
                                ) -> None:
        """
        Creates a `PrimitiveState` object from a `ConservativeState` object. Given that `PrimitiveState` state
        vector is $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$ and `ConservativeState` state vector is
        $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, and labelling the components of $W$ as
        $\\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\end{bmatrix}^T$ and the components of $U$ as
        $\\begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \\end{bmatrix}^T$, then, the transformation from $U$ to $W$ is: \n
        $\\begin{bmatrix} w_1 \\\\\ w_2 \\\\\ w_3 \\\\\ w_4 \\end{bmatrix} =
        \\begin{bmatrix} u_1 \\\\\ u_2 / u_1 \\\\\ u_3 / u_1 \\\\\ (\\gamma - 1)[u_4 - \\frac{1}{2 \\rho}(u_2^2 + u_3^2)] \\\\\ \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u / \\rho \\\\\ \\rho v / \\rho \\\\\ (\\gamma - 1)[e - \\frac{1}{2 \\rho}((\\rho u)^2 + (\\rho v)^2)] \\\\\ \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ u \\\\\ v \\\\\ p \\end{bmatrix}$

        **Parameters**                              \n
            U       ConservativeState object        \n
        """
        self.q0 = U.rho.copy()
        self.q1 = U.q1 / self.q0
        self.q2 = U.q2 / self.q0
        self.q3 = (self.g - 1) * (U.q3 - self.ek())

        self.set_state_from_vars()


    def from_conservative_state_vector(self,
                                       U_vector: np.ndarray
                                       ) -> None:
        """

        """
        self.q0 = U_vector[:, :, self.RHO_IDX].copy()
        self.q1 = U_vector[:, :, self.U_IDX] / self.q0
        self.q2 = U_vector[:, :, self.V_IDX] / self.q0
        self.q3 = (self.g - 1) * (U_vector[:, :, self.P_IDX] - self.ek())

        self.set_state_from_vars()

    def from_conservative_state_vars(self,
                                     rho: np.ndarray,
                                     rhou: np.ndarray,
                                     rhov: np.ndarray,
                                     e: np.ndarray
                                     ) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components based on conservative
        state variables

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        self.q0 = rho.copy()
        self.q1 = rhou / rho
        self.q2 = rhov / rho
        self.q3 = (self.g - 1) * (e - 0.5 * (rhou * rhou + rhov * rhov) / rho)

        # Set W components appropriately
        self.set_state_from_vars()

    def from_primitive_state(self,
                             W: 'PrimitiveState'
                             ) -> None:
        """

        """
        self.W = W.W.copy()
        self.set_vars_from_state()

    def from_primitive_state_vector(self,
                                    W_vector: np.ndarray
                                    ) -> None:
        """

        """
        self.W = W_vector.copy()
        self.set_vars_from_state()

    def from_primitive_state_vars(self,
                                  rho: np.ndarray,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  p: np.ndarray,
                                  copy: bool = False,
                                  ) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components based on primitive state
        variables

        **Parameters**                              \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Set density, u, v and pressure
        if copy:
            self.q0 = rho.copy()
            self.q1 = u.copy()
            self.q2 = v.copy()
            self.q3 = p.copy()
        else:
            self.q0 = rho
            self.q1 = u
            self.q2 = v
            self.q3 = p

        # Set W components appropriately
        self.set_state_from_vars()
        
    def from_primitive_to_conservative(self, ar: np.ndarray):
        ar[:, :, 0], ar[:, :, 1], ar[:, :, 2], ar[:, :, 3] = \
         ar[:, :, 0], ar[:, :, 1] * ar[:, :, 0], ar[:, :, 2] * ar[:, :, 0], \
         ar[:, :, 3] / (self.g - 1) + self.ek_JIT(ar[:, :, 0], ar[:, :, 1], ar[:, :, 2])
        return ar

    def to_conservative_state(self) -> 'ConservativeState':
        """
        Creates a `ConservativeState` object from itself
        """
        return ConservativeState(self.inputs, W_vector=self.W)

    def to_conservative_vector(self) -> np.ndarray:

        U = np.zeros_like(self.W)

        U[:, :, ConservativeState.RHO_IDX] = self.rho.copy()
        U[:, :, ConservativeState.RHOU_IDX] = self.q1 * self.q0
        U[:, :, ConservativeState.RHOV_IDX] = self.q2 * self.q0
        U[:, :, ConservativeState.E_IDX] = self.q3 / (self.g - 1) + self.ek()

        return U

    #@cache
    def ek(self) -> np.ndarray:
        return self.ek_JIT(self.q0, self.q1, self.q2)


    @staticmethod
    @nb.njit(cache=True)
    def ek_JIT(rho: np.ndarray,
               u: np.ndarray,
               v: np.ndarray,
               ) -> np.ndarray:
        _Ek = np.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                _Ek[i, j] = 0.5 * rho[i, j] * (u[i, j] * u[i, j] + v[i, j] * v[i, j])
        return _Ek

    #@cache
    def Ek(self) -> np.ndarray:
        return self.Ek_JIT(self.q1, self.q2)

    @staticmethod
    @nb.njit(cache=True)
    def Ek_JIT(u: np.ndarray,
               v: np.ndarray
               ) -> np.ndarray:
        _Ek = np.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                _Ek[i, j] = 0.5 * (u[i, j] * u[i, j] + v[i, j] * v[i, j])
        return _Ek

    #@cache
    def H(self,
          Ek: np.ndarray = None
          ) -> np.ndarray:
        if Ek is None:
            return self.H_JIT(self.q0, self.q1, self.q2, self.q3, self.g_over_gm)
        else:
            return self.H_given_Ek_JIT(self.q0, self.q3, Ek, self.g_over_gm)


    @staticmethod
    @nb.njit(cache=True)
    def H_minus_Ek_JIT(rho: np.ndarray,
                       p: np.ndarray,
                       gm: float,
                       ) -> np.ndarray:
        _H = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _H[i, j] = gm * p[i, j] / rho[i, j]
        return _H

    @staticmethod
    @nb.njit(cache=True)
    def H_given_Ek_JIT(rho: np.ndarray,
                       p: np.ndarray,
                       Ek: np.ndarray,
                       gm: float,
                       ) -> np.ndarray:
        _H = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _H[i, j] = gm * p[i, j] / rho[i, j] + Ek[i, j]
        return _H

    @staticmethod
    @nb.njit(cache=True)
    def H_JIT(rho: np.ndarray,
              u: np.ndarray,
              v: np.ndarray,
              p: np.ndarray,
              gm: float,
              ) -> np.ndarray:
        _H = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _H[i, j] = gm * p[i, j] / rho[i, j] + 0.5 * (u[i, j] * u[i, j] + v[i, j] * v[i, j])
        return _H

    #@cache
    def a(self) -> np.ndarray:
        #return np.sqrt(self.g * self.q3 / self.q0)
        return self.a_JIT(self.q3, self.q0, self.g)

    @staticmethod
    @nb.njit(cache=True)
    def a_JIT(p: np.ndarray,
              rho: np.ndarray,
              g: float,
              ) -> np.ndarray:
        _a = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _a[i, j] = np.sqrt(g * p[i, j] / rho[i, j])
        return _a

    def e(self):
        return self.q3 / (self.g - 1) + self.ek()

    def V(self) -> np.ndarray:
        return np.sqrt(self.q1 * self.q1 + self.q2 * self.q2)

    def Ma(self) -> np.ndarray:
        return self.V() / self.a()

    def non_dim(self) -> None:
        """
        Non-dimentionalizes the primitive state vector W. Let the non-dimentionalized primitive state vector be
        $W^* = \\begin{bmatrix} \\rho^* \\ u^* \\ v^* \\ p^* \\end{bmatrix}^T$, where $^*$ indicates a
        non-dimentionalized quantity. The non-dimentionalized primitive variables are:  \n
        $\\rho^* = \\rho/\\rho_\\infty$                                                 \n
        $u^* = u/a_\\infty$                                                             \n
        $u^* = v/a_\\infty$                                                             \n
        $p^* = p/\\rho_\\infty a_\\infty^2$                                             \n
        If `PrimitiveState` is created from a non-dimentionalized `ConservativeState`, it will be non-dimentional.
        """

        # Non-dimentionalize each component of W
        self.W[:, :, ConservativeState.RHO_IDX]     /= self.inputs.rho_inf
        self.W[:, :, ConservativeState.RHOU_IDX]    /= self.inputs.rho_inf * self.inputs.a_inf
        self.W[:, :, ConservativeState.RHOV_IDX]    /= self.inputs.rho_inf * self.inputs.a_inf
        self.W[:, :, ConservativeState.E_IDX]       /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        # Set variables from non-dimensionalized W
        self.set_vars_from_state()


    def F(self,
          U: ConservativeState = None,
          U_vector: np.ndarray = None) -> np.ndarray:

        F = np.zeros_like(self.W)

        if U is not None:
            F[:, :, 0] = U.rhou
            F[:, :, 1] = U.rhou * self.u + self.p
            F[:, :, 2] = U.rhou * self.v
            F[:, :, 3] = self.u * (U.e + self.p)
            return F

        elif U_vector is not None:
            ru = U_vector[:, :, ConservativeState.RHOU_IDX]
            e = U_vector[:, :, ConservativeState.E_IDX]

            F[:, :, 0] = ru
            F[:, :, 1] = ru * self.u + self.p
            F[:, :, 2] = ru * self.v
            F[:, :, 3] = self.u * (e + self.p)
            return F

        else:
            ru = self.rho * self.u

            F[:, :, 0] = self.rho * self.u
            F[:, :, 1] = ru * self.u + self.p
            F[:, :, 2] = ru * self.v
            F[:, :, 3] = self.u * (self.e() + self.p)
            return F


class ConservativeState(State):
    """
    #Conservative Solution State#
    A class that represents the solution state vector of the 2D inviscid Euler equations in conservative form. The
    conservative solution vector, $U$, is represented mathematically as:
    $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, where $\\rho$ is the solution density,
    $u$ is the $x$-direction momentum per unit volume, $v$ is the $y$-direction momentum per unit volume, and $e$ is
    the energy per unit volume. `ConservativeState` can be used to represent the solution state of a QuadBlock in a
    solver that utilizes a conservative formulation. It can also be used to represent the solution state in
    BoundaryBlocks in a solver that utilizes a conservative formulation.
    """

    RHO_IDX     = 0
    RHOU_IDX    = 1
    RHOV_IDX    = 2
    E_IDX       = 3

    def __init__(self,
                 inputs: ProblemInput,
                 nx: int = None,
                 ny: int = None,
                 state: State = None,
                 U_vector: np.ndarray = None,
                 W_vector: np.ndarray = None):
        """
        ## Attributes

        **Public**                                                  \n
            U       Conservative state vector                       \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Check if an input vector is given
        if state or (U_vector is not None) or (W_vector is not None):
            _vec_given = True
        else:
            _vec_given = False

        # Check if more than one input vector is given
        if _vec_given and sum(map(bool, [state, U_vector is not None, W_vector is not None])) > 1:
            raise ValueError('Please provide only one type of input state or vector')

        # Get vector size
        _nx, _ny = 0, 0

        if ny and nx and not _vec_given:
            _nx, _ny = int(nx), int(ny)
        elif _vec_given:
            if state:
                _nx, _ny = state.nx, state.ny
            elif U_vector is not None:
                _nx, _ny = int(U_vector.shape[1]), int(U_vector.shape[0])
            elif W_vector is not None:
                _nx, _ny = int(W_vector.shape[1]), int(W_vector.shape[0])
        else:
            ValueError('Please either provide the size of the vector or a state class or a state vector')

        super().__init__(inputs, _nx, _ny)

        if state:
            if isinstance(state, PrimitiveState):
                self.from_primitive_state(state)
            elif isinstance(state, ConservativeState):
                self.from_conservative_state(state)

        elif U_vector is not None:
            self.from_conservative_state_vector(U_vector)

        elif W_vector is not None:
            self.from_primitive_state_vector(W_vector)

    # ------------------------------------------------------------------------------------------------------------------
    # Define primitive state properties

    # rho property (refers to q0 in Q)
    @property
    def rho(self) -> np.ndarray:
        return self.q0

    @rho.setter
    def rho(self,
            rho: np.ndarray
            ) -> None:
        self.q0 = rho

    # u property (refers to q1 in Q)
    @property
    def rhou(self) -> np.ndarray:
        return self.q1

    @rhou.setter
    def rhou(self,
             rhou: np.ndarray
             ) -> None:
        self.q1 = rhou

    # v property (refers to q2 in Q)
    @property
    def rhov(self) -> np.ndarray:
        return self.q2

    @rhov.setter
    def rhov(self,
             rhov: np.ndarray
             ) -> None:
        self.q2 = rhov

    # e property (refers to q3 in Q)
    @property
    def e(self) -> np.ndarray:
        return self.q3

    @e.setter
    def e(self, e: np.ndarray) -> None:
        self.q3 = e

    # W property (refers to q3 in Q)
    @property
    def U(self) -> np.ndarray:
        return self.Q

    # W property (refers to q3 in Q)
    @U.setter
    def U(self, U: np.ndarray):
        self.Q = U

    # PUBLIC METHODS ---------------------------------------------------------------------------------------------------

    def from_conservative_state(self,
                                U: 'ConservativeState'
                                ) -> None:
        self.U = U.U.copy()
        self.set_vars_from_state()

    def from_conservative_state_vector(self,
                                       U_vector: np.ndarray,
                                       copy: bool = True,
                                       ) -> None:

        self.U = U_vector.copy() if copy else U_vector
        self.set_vars_from_state()

    def from_conservative_state_vars(self,
                                     rho: np.ndarray,
                                     rhou: np.ndarray,
                                     rhov: np.ndarray,
                                     e: np.ndarray
                                     ) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                                              \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Set density, u, v and pressure
        self.q0     = rho.copy()
        self.q1     = rhou.copy()
        self.q2     = rhov.copy()
        self.q3     = e.copy()

        # Set state vector from state variables
        self.set_state_from_vars()

    def from_primitive_state(self,
                             W: PrimitiveState
                             ) -> None:
        """
        Creates a `COnservativeState` object from a `PrimitiveState` object. Given that `PrimitiveState` state
        vector is $W = \\begin{bmatrix} \\rho \\ u \\ v \\ p \\end{bmatrix}^T$ and `ConservativeState` state vector is
        $U = \\begin{bmatrix} \\rho \\ \\rho u \\ \\rho v \\ e \\end{bmatrix}^T$, and labelling the components of $W$ as
        $\\begin{bmatrix} w_1 \\ w_2 \\ w_3 \\ w_4 \\end{bmatrix}^T$ and the components of $U$ as
        $\\begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \\end{bmatrix}^T$, then, the transformation from $U$ to $W$ is: \n
        $\\begin{bmatrix} u_1 \\\\\ u_2 \\\\\ u_3 \\\\\ u_4 \\end{bmatrix} =
        \\begin{bmatrix} w_1 \\\\\ w_2w_1 \\\\\ w_3w_1 \\\\\ \\frac{w_4}{\\gamma - 1} + \\frac{w_1}{2}(w_2^2 + w_3^2) \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u \\\\\ \\rho v \\\\\ \\frac{p}{\\gamma - 1} + \\frac{\\rho}{2} (u^2 + v^2) \\end{bmatrix} =
        \\begin{bmatrix} \\rho \\\\\ \\rho u \\\\\ \\rho v \\\\\ e \\end{bmatrix}$

        **Parameters**                              \n
            W       PrimitiveState object           \n
        """
        self.q0     = W.rho.copy()
        self.q1     = W.rho * W.u
        self.q2     = W.rho * W.v
        self.q3     = W.p / (self.g - 1) + W.ek()

        self.set_state_from_vars()

    def from_primitive_state_vector(self,
                                    W_vector: np.ndarray
                                    ) -> None:
        """

        """
        self.q0     = W_vector[:, :, PrimitiveState.RHO_IDX].copy()
        self.q1     = W_vector[:, :, PrimitiveState.U_IDX] * self.q0
        self.q2     = W_vector[:, :, PrimitiveState.V_IDX] * self.q0
        self.q3     = W_vector[:, :, PrimitiveState.P_IDX] / (self.g - 1) + self.ek()

        self.set_state_from_vars()

    def from_primitive_state_vars(self,
                                  rho: np.ndarray,
                                  u: np.ndarray,
                                  v: np.ndarray,
                                  p: np.ndarray
                                  ) -> None:
        """
        Populates the primitive state vector and variables W with their appropriate components

        **Parameters**                                              \n
            rho     density                                         \n
            rhou    x-direction momentum per unit volume            \n
            rhov    y-direction momentum per unit volume            \n
            e       energy per unit volume                          \n
        """

        # Set density, u, v and pressure
        self.q0     = rho.copy()
        self.q1     = rho * u
        self.q2     = rho * v
        self.q3     = p / (self.g - 1) + 0.5 * rho * (u * u + v * v)

        # Set state vector from state variables
        self.set_state_from_vars()

    def to_primitive_state(self) -> PrimitiveState:
        """
        Creates a `PrimitiveState` object from itself
        """
        return PrimitiveState(self.inputs, U_vector=self.U)

    def to_primitive_vector(self):

        W = np.zeros_like(self.U)

        W[:, :, PrimitiveState.RHO_IDX] = self.rho.copy()
        W[:, :, PrimitiveState.U_IDX] = self.u()
        W[:, :, PrimitiveState.V_IDX] = self.v()
        W[:, :, PrimitiveState.P_IDX] = self.p()

        return W

    def u(self) -> np.ndarray:
        return self.q1 / self.q0

    def v(self) -> np.ndarray:
        return self.q2 / self.q0

    def ek(self) -> np.ndarray:
        return 0.5 * (self.q1 * self.q1 + self.q2 * self.q2) / self.q0

    def h(self) -> np.ndarray:
        _ek = self.ek()
        return self.g * (self.q3 - _ek) + _ek

    def H(self) -> np.ndarray:
        return self.h() / self.q0

    def a(self) -> np.ndarray:
        return np.sqrt(self.g * self.p() / self.q0)

    def p(self) -> np.ndarray:
        return (self.g - 1) * (self.q3 - self.ek())

    def V(self) -> np.ndarray:
        return np.sqrt(self.u() ** 2 + self.v() ** 2)

    def Ma(self) -> np.ndarray:
        return self.V() / self.a()

    def non_dim(self) -> None:
        """
        Non-dimentionalizes the conservative state vector W. Let the non-dimentionalized conservative state vector be
        $U^* = \\begin{bmatrix} \\rho^* \\ \\rho u^* \\ \\rho v^* \\ e^* \\end{bmatrix}^T$, where $^*$ indicates a
        non-dimentionalized quantity. The non-dimentionalized conservative variables are:   \n
        $\\rho^* = \\rho/\\rho_\\infty$                                                     \n
        $\\rho u^* = \\rho u/a_\\infty \\rho_\\infty$                                       \n
        $\\rho u^* = \\rho v/a_\\infty \\rho_\\infty$                                       \n
        $e^* = e/\\rho_\\infty a_\\infty^2$                                                 \n
        If `ConservativeState` is created from a non-dimentionalized `PrimitiveState`, it will be non-dimentional.
        """

        self.U[:, :, self.RHO_IDX]  /= self.inputs.rho_inf
        self.U[:, :, self.RHOU_IDX] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[:, :, self.RHOV_IDX] /= self.inputs.rho_inf * self.inputs.a_inf
        self.U[:, :, self.E_IDX]    /= self.inputs.rho_inf * self.inputs.a_inf ** 2

        self.set_vars_from_state()

    def F(self) -> np.ndarray:

        u = self.u()
        ru = self.rho * u
        p = self.p()

        return np.dstack((ru,
                          ru * u + p,
                          ru * self.v(),
                          u * (self.e + p)))

    def G(self) -> np.ndarray:

        v = self.v()
        rv = self.rho * v
        p = self.p()

        return np.dstack((rv,
                          rv * self.u(),
                          rv * v + p,
                          v * (self.e + p)))


class RoePrimitiveState(PrimitiveState):
    """
    #*$Roe$* Primitive State
    A class the represents the *Roe* average state in primitive form. This class extends `PrimitiveState`, and is used to
    compute the *Roe* average of two primitive states, usually the left and right solution vectors $(W^R, W^L)$ at the
    left and right cell boundaries in a finite volume method. The primitive *Roe* average state vector is
    $W^* = \\begin{bmatrix} \\rho^* \\ u^* \\ v^* \\ p^* \\end{bmatrix}^T$, where $\\rho^\\*$ is the *Roe* density,
    $u^\\*$ is the $x$-direction *Roe* momentum per unit volume, $v^\\*$ is the $y$-direction *Roe* momentum per unit volume,
    and $p^\\*$ is the *Roe* energy per unit volume. This directly represents the *Roe* average quantities that can be
    used in various approximate Riemann solvers such as *Roe, HLLE, HLLL,* etc. The *Roe* average is computed as follows: \n

    Given the left and right primitive solution vectors $(W^R, W^L)$, where
    $W^R = \\begin{bmatrix} \\rho^R \\ u^R \\ v^R \\ p^R \\end{bmatrix}^T$ and
    $W^L = \\begin{bmatrix} \\rho^L \\ u^L \\ v^L \\ p^L \\end{bmatrix}^T$, the components of $W^\\*$ are:  \n
    $\\rho^\\* = \\sqrt{\\rho^R \\rho^L}$,                                                                  \n
    $u^\\* = \\frac{u^R \\sqrt{\\rho^R} + u^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$,         \n
    $v^\\* = \\frac{v^R \\sqrt{\\rho^R} + v^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$,         \n
    $p^\\* = \\frac{\\gamma - 1}{\\gamma}\\rho^\\*[H^\\* - \\frac{1}{2}({u^\\*}^2 + {v^\\*}^2)]$,           \n
    where $H^\\*$ is the Roe specific enthalpy, evaluated as:                                               \n
    $H^\\* = \\frac{H^R \\sqrt{\\rho^R} + H^L \\sqrt{\\rho^L}}{\\sqrt{\\rho^R} + \\sqrt{\\rho^L}}$.
    """

    def __init__(self,
                 inputs: ProblemInput,
                 WL: PrimitiveState = None,
                 WR: PrimitiveState = None,
                 UL: ConservativeState = None,
                 UR: ConservativeState = None
                 ):
        """
        ## Attributes

        **Public**                                  \n
            W       Primitive state vector          \n
            rho     density                         \n
            u       x-direction velocity            \n
            v       y-direction velocity            \n
            p       pressure                        \n
        """

        # Call superclass constructor
        if WL and WR:
            if UL or UR:
                raise ValueError('Cannot provide ConservativeStates along with PrimitiveStates.')
            else:
                super().__init__(inputs, nx=WL.W.shape[1], ny=WL.W.shape[0])
        elif UL and UR:
            if WL or WR:
                raise ValueError('Cannot provide PrimitiveStates along with ConservativeStates.')
            else:
                super().__init__(inputs, nx=UL.U.shape[1], ny=UL.U.shape[0])

        self.roe_state_from_primitive_states(WL, WR)


    def roe_state_from_primitive_states(self,
                                        WL: PrimitiveState,
                                        WR: PrimitiveState
                                        ) -> None:
        """
        Compute *Roe* average quantities of left and right primtive states

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        # Compute common quantities
        sqRhoL  = np.sqrt(WL.q0)
        sqRhoR  = np.sqrt(WR.q0)
        sqRhoRL = sqRhoL + sqRhoR

        # Compute *Roe* average quantities
        self.q0 = np.sqrt(WL.q0 * WR.q0)
        self.q1 = (WL.q1 * sqRhoL + WR.q1 * sqRhoR) / sqRhoRL
        self.q2 = (WL.q2 * sqRhoL + WR.q2 * sqRhoR) / sqRhoRL
        e       = (WL.e() * sqRhoL + WR.e() * sqRhoR) / sqRhoRL
        self.q3 = (self.g - 1) * (e - self.ek())

        # Set state vector and variables from *Roe* averages
        self.set_state_from_vars()


    def roe_state_from_conservative_states(self,
                                           UL: ConservativeState,
                                           UR: ConservativeState
                                           ) -> None:
        """
         Compute *Roe* average quantities of left and right conservative states

         **Parameters**                                              \n
             UL      Left conservative state                         \n
             UR      Right conservative state                        \n
         """

        # Get average *Roe* quantities
        sqRhoL = np.sqrt(UL.q0)
        sqRhoR = np.sqrt(UR.q0)
        sqRhoRL = sqRhoL + sqRhoR

        # Compute *Roe* average quantities
        rho = np.sqrt(UL.q0 * UR.q0)
        u = (UL.q1 / sqRhoL + UR.q1 / sqRhoR) / sqRhoRL
        v = (UL.q2 / sqRhoL + UR.q2 / sqRhoR) / sqRhoRL
        H = (UL.H() * sqRhoL + UR.H() * sqRhoR) / sqRhoRL
        p = (self.g - 1) / self.g * rho * (H - 0.5 * (u ** 2 + v ** 2))

        # Set state vector and variables from *Roe* averages
        self.from_primitive_state_vars(rho, u, v, p, copy=False)
