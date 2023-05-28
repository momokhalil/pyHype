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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numba as nb
import numpy as np
from pyhype.states.base import State
from pyhype.states.conservative import ConservativeState
from pyhype.states.converter import PrimitiveConverter
from pyhype.utils.utils import cache

from typing import TYPE_CHECKING, Union, Type

if TYPE_CHECKING:
    from pyhype.fluids.base import Fluid


class PrimitiveState(State):
    """
    A class that represents the solution state vector of the 2D inviscid Euler equations in primitive form.
    """

    RHO_IDX = 0
    U_IDX = 1
    V_IDX = 2
    P_IDX = 3

    def __init__(
        self,
        fluid: Fluid,
        state: State = None,
        array: np.ndarray = None,
        shape: tuple[int, int] = None,
        fill: Union[float, int] = None,
    ):
        super().__init__(fluid=fluid, state=state, array=array, shape=shape, fill=fill)

    def get_class_type_converter(self) -> Type[PrimitiveConverter]:
        return PrimitiveConverter

    @property
    def rho(self) -> np.ndarray:
        return self._data[:, :, self.RHO_IDX]

    @rho.setter
    def rho(self, rho: np.ndarray) -> None:
        self._data[:, :, self.RHO_IDX] = rho

    @property
    def u(self) -> np.ndarray:
        return self._data[:, :, self.U_IDX]

    @u.setter
    def u(self, u: np.ndarray) -> None:
        self._data[:, :, self.U_IDX] = u

    @property
    def v(self) -> np.ndarray:
        return self._data[:, :, self.V_IDX]

    @v.setter
    def v(self, v: np.ndarray) -> None:
        self._data[:, :, self.V_IDX] = v

    @property
    def p(self) -> np.ndarray:
        return self._data[:, :, self.P_IDX]

    @p.setter
    def p(self, p: np.ndarray) -> None:
        self._data[:, :, self.P_IDX] = p

    @cache
    def ek(self) -> np.ndarray:
        return self.ek_JIT(self._data)

    def ek_NP(self):
        return self.rho * self.Ek_NP()

    @staticmethod
    @nb.njit(cache=True)
    def ek_JIT(W: np.ndarray) -> np.ndarray:
        _ek = np.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                _ek[i, j] = (
                    0.5
                    * W[i, j, 0]
                    * (W[i, j, 1] * W[i, j, 1] + W[i, j, 2] * W[i, j, 2])
                )
        return _ek

    @cache
    def Ek(self) -> np.ndarray:
        return self.Ek_JIT(self._data)

    def Ek_NP(self):
        _u = self.u
        _v = self.v
        return 0.5 * (_u * _u + _v * _v)

    @staticmethod
    @nb.njit(cache=True)
    def Ek_JIT(W: np.ndarray) -> np.ndarray:
        _Ek = np.zeros((W.shape[0], W.shape[1]))
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                _Ek[i, j] = 0.5 * (W[i, j, 1] * W[i, j, 1] + W[i, j, 2] * W[i, j, 2])
        return _Ek

    @cache
    def H(self, Ek: np.ndarray = None) -> np.ndarray:
        if Ek is None:
            return self.H_JIT(self.rho, self.u, self.v, self.p, self.fluid.g_over_gm1())
        return self.H_given_Ek_JIT(self.rho, self.p, Ek, self.fluid.g_over_gm1())

    @staticmethod
    @nb.njit(cache=True)
    def H_minus_Ek_JIT(
        rho: np.ndarray,
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
    def H_given_Ek_JIT(
        rho: np.ndarray,
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
    def H_JIT(
        rho: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        p: np.ndarray,
        gm: float,
    ) -> np.ndarray:
        _H = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _H[i, j] = gm * p[i, j] / rho[i, j] + 0.5 * (
                    u[i, j] * u[i, j] + v[i, j] * v[i, j]
                )
        return _H

    @cache
    def a(self) -> np.ndarray:
        return self.a_JIT(self.p, self.rho, self.fluid.gamma())

    @staticmethod
    @nb.njit(cache=True)
    def a_JIT(
        p: np.ndarray,
        rho: np.ndarray,
        g: float,
    ) -> np.ndarray:
        _a = np.zeros_like(p)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                _a[i, j] = np.sqrt(g * p[i, j] / rho[i, j])
        return _a

    @cache
    def e(self):
        return self.fluid.one_over_gm1() * self.p + self.ek()

    def V(self) -> np.ndarray:
        return np.sqrt(self.u * self.u + self.v * self.v)

    def Ma(self) -> np.ndarray:
        return self.V() / self.a()

    def F(self, U: ConservativeState = None, U_vector: np.ndarray = None) -> np.ndarray:

        if U is not None:
            F = np.zeros_like(self.data, dtype=float)
            F[:, :, 0] = U.rhou
            F[:, :, 1] = U.rhou * self.u + self.p
            F[:, :, 2] = U.rhou * self.v
            F[:, :, 3] = self.u * (U.e + self.p)
            return F

        if U_vector is not None:
            F = np.zeros_like(self.data, dtype=float)
            ru = U_vector[:, :, ConservativeState.RHOU_IDX]

            F[:, :, 0] = ru
            F[:, :, 1] = ru * self.u + self.p
            F[:, :, 2] = ru * self.v
            F[:, :, 3] = self.u * (U_vector[:, :, ConservativeState.E_IDX] + self.p)
            return F

        return self._F_from_prim_JIT(self._data, self.ek(), self.fluid.one_over_gm1())

    @staticmethod
    @nb.njit(cache=True)
    def _F_from_prim_JIT(W, ek, k):
        _F = np.zeros((W.shape[0], W.shape[1], 4))
        _u = 0.0
        _ru = 0.0
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                _u = W[i, j, 1]
                _ru = W[i, j, 0] * _u
                _F[i, j, 0] = _ru
                _F[i, j, 1] = _ru * _u + W[i, j, 3]
                _F[i, j, 2] = _ru * W[i, j, 2]
                _F[i, j, 3] = _u * (k * W[i, j, 3] + ek[i, j] + W[i, j, 3])
        return _F

    def realizability_conditions(self) -> dict[str, np.ndarray]:
        return dict(
            rho_good=self.rho > 0,
            pressure_good=self.p > 0,
        )


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

    def __init__(
        self,
        fluid: Fluid,
        WL: PrimitiveState,
        WR: PrimitiveState,
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

        roe_state_array = self.roe_state_from_primitive_states(WL, WR)
        super().__init__(fluid=fluid, array=roe_state_array)

    def roe_state_from_primitive_states(
        self, WL: PrimitiveState, WR: PrimitiveState
    ) -> np.ndarray:
        """
        Compute *Roe* average quantities of left and right primtive states

        **Parameters**                                              \n
            WL      Left primitive state                            \n
            WR      Right primitive state                           \n
        """

        return self._roe_state_from_prim_JIT(WL.data, WR.data)

    @staticmethod
    @nb.njit(cache=True)
    def _roe_state_from_prim_JIT(QL: np.ndarray, QR: np.ndarray) -> np.ndarray:
        _Q = np.zeros_like(QL)
        for i in range(QL.shape[0]):
            for j in range(QL.shape[1]):
                sqRhoL = np.sqrt(QL[i, j, 0])
                sqRhoR = np.sqrt(QR[i, j, 0])
                sqRhoRL_inv = 1.0 / (sqRhoL + sqRhoR)
                _Q[i, j, 0] = np.sqrt(QL[i, j, 0] * QR[i, j, 0])
                _Q[i, j, 1] = (
                    QL[i, j, 1] * sqRhoL + QR[i, j, 1] * sqRhoR
                ) * sqRhoRL_inv
                _Q[i, j, 2] = (
                    QL[i, j, 2] * sqRhoL + QR[i, j, 2] * sqRhoR
                ) * sqRhoRL_inv
                _Q[i, j, 3] = (
                    QL[i, j, 3] * sqRhoL + QR[i, j, 3] * sqRhoR
                ) * sqRhoRL_inv
        return _Q
