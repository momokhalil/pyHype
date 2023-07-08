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

import numpy as np
from pyhype.states.base import State
from pyhype.utils.utils import cache

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from pyhype.fluids.base import Fluid


class ConservativeState(State):
    """
    A class that represents the solution state vector of the 2D inviscid Euler equations in conservative form.
    """

    RHO_IDX = 0
    RHOU_IDX = 1
    RHOV_IDX = 2
    E_IDX = 3

    def __init__(
        self,
        fluid: Fluid,
        state: State = None,
        array: np.ndarray = None,
        shape: tuple[int, int] = None,
        fill: Union[float, int] = None,
    ):

        super().__init__(fluid=fluid, state=state, array=array, shape=shape, fill=fill)

    @property
    def rho(self) -> np.ndarray:
        return self._data[:, :, self.RHO_IDX]

    @rho.setter
    def rho(self, rho: np.ndarray) -> None:
        self._data[:, :, self.RHO_IDX] = rho

    @property
    def rhou(self) -> np.ndarray:
        return self._data[:, :, self.RHOU_IDX]

    @rhou.setter
    def rhou(self, rhou: np.ndarray) -> None:
        self._data[:, :, self.RHOU_IDX] = rhou

    @property
    def rhov(self) -> np.ndarray:
        return self._data[:, :, self.RHOV_IDX]

    @rhov.setter
    def rhov(self, rhov: np.ndarray) -> None:
        self._data[:, :, self.RHOV_IDX] = rhov

    @property
    def e(self) -> np.ndarray:
        return self._data[:, :, self.E_IDX]

    @e.setter
    def e(self, e: np.ndarray) -> None:
        self._data[:, :, self.E_IDX] = e

    @property
    @cache
    def u(self) -> np.ndarray:
        return self._data[:, :, self.RHOU_IDX] / self.rho

    @u.setter
    def u(self, u: np.ndarray) -> None:
        raise NotImplementedError(
            'Property "u" is not settable for class' + str(type(self))
        )

    @property
    @cache
    def v(self) -> np.ndarray:
        return self._data[:, :, self.RHOV_IDX] / self.rho

    @v.setter
    def v(self, v: np.ndarray) -> None:
        raise NotImplementedError(
            'Property "v" is not settable for class' + str(type(self))
        )

    @property
    @cache
    def p(self) -> np.ndarray:
        return (self.fluid.gamma() - 1) * (self.e - self.ek())

    @p.setter
    def p(self, p: np.ndarray) -> None:
        raise NotImplementedError(
            'Property "p" is not settable for class' + str(type(self))
        )

    @cache
    def ek(self) -> np.ndarray:
        return self.rho * self.Ek()

    @cache
    def Ek(self) -> np.ndarray:
        return 0.5 * (self.u * self.u + self.v * self.v)

    @cache
    def h(self) -> np.ndarray:
        _ek = self.ek()
        return self.fluid.gamma() * (self.e - _ek) + _ek

    @cache
    def H(self) -> np.ndarray:
        return self.h() / self.rho

    @cache
    def a(self) -> np.ndarray:
        return np.sqrt(self.fluid.gamma() * self.p / self.rho)

    def V(self) -> np.ndarray:
        return np.sqrt(self.u**2 + self.v**2)

    def Ma(self) -> np.ndarray:
        return self.V() / self.a()

    def F(self) -> np.ndarray:

        u = self.u
        ru = self.rho * u
        p = self.p

        return np.dstack((ru, ru * u + p, ru * self.v, u * (self.e + p)))

    def G(self) -> np.ndarray:

        v = self.v
        rv = self.rho * v
        p = self.p

        return np.dstack((rv, rv * self.u, rv * v + p, v * (self.e + p)))

    def realizability_conditions(self) -> dict[str, np.ndarray]:
        return dict(
            rho_good=self.rho > 0,
            energy_good=self.e > 0,
        )
