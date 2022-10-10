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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, overload

import numpy as np

if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput

StateCompatible = (int, float, np.ndarray)


class State(ABC):
    """
    # State
    Defines an abstract class for implementing primitive and conservative state classes. The core components of a state
    are the state vector and the state variables. The state vector is composed of the state variables in a specific
    order. For example, for a state X with state variables $x_1, x_2, ..., x_n$ and state vector $X$, the state vector
    is represented as:
    $X = \\begin{bmatrix} x_1 \\ x_2 \\ \\dots \\ x_n \\end{bmatrix}^T$. The state vector represents the solution at
    each physical discretization point.
    """

    converter = None

    def __init__(
        self,
        inputs: ProblemInput,
        state: State = None,
        array: np.ndarray = None,
        shape: tuple[int, int] = None,
        fill: Union[float, int] = None,
    ):
        """
        ## Attributes

        **Private**                                 \n
            input       input dictionary            \n
            size        size of grid in block       \n

        **Public**                                  \n
            g           (gamma) specific heat ratio \n
        """

        self.inputs = inputs
        self.g = inputs.gamma
        self.g_over_gm = self.g / (self.g - 1)
        self.one_over_gm = 1 / (self.g - 1)

        if state is not None:
            self.from_state(state)
        elif array is not None:
            if array.ndim != 3 or array.shape[-1] != 4:
                raise ValueError("Array must have 3 dims and a depth of 4.")
            self._Q = array
        elif shape is not None:
            if fill is not None:
                self._Q = np.full(shape=shape, fill_value=fill)
            else:
                self._Q = np.zeros(shape=shape)
        else:
            self._Q = np.zeros((0, 0))

        self.cache = {}

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
        if not isinstance(Q, np.ndarray):
            raise TypeError(
                f"Input array must be a numpy ndarray, but it is a {type(Q)}."
            )
        self._Q = Q
        self.clear_cache()

    @property
    @abstractmethod
    def rho(self) -> None:
        raise NotImplementedError

    @rho.setter
    @abstractmethod
    def rho(self, rho: np.ndarray) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def u(self) -> None:
        raise NotImplementedError

    @u.setter
    @abstractmethod
    def u(self, u: np.ndarray) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def v(self) -> None:
        raise NotImplementedError

    @v.setter
    @abstractmethod
    def v(self, v: np.ndarray) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def p(self) -> None:
        raise NotImplementedError

    @p.setter
    @abstractmethod
    def p(self, p: np.ndarray) -> None:
        raise NotImplementedError

    def __getitem__(self, index: Union[int, slice]) -> State:
        return type(self)(self.inputs, array=self.Q[index].copy())

    def __add__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=self.Q + other.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=self.Q + other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __radd__(self, other: Union[State, StateCompatible]) -> State:
        return self.__add__(other)

    def __sub__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=self.Q - other.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=self.Q - other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rsub__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=other.Q - self.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=other - self.Q)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __mul__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=self.Q * other.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=self.Q * other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rmul__(self, other: Union[State, StateCompatible]):
        return self.__mul__(other)

    def __truediv__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=self.Q / other.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=self.Q / other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rtruediv__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=other.Q / self.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=other / self.Q)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __pow__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=self.Q**other.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=self.Q**other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rpow__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(inputs=self.inputs, array=other.Q**self.Q)
        if isinstance(other, StateCompatible):
            return type(self)(inputs=self.inputs, array=other**self.Q)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def reset(self, shape: tuple[int] = None):
        if shape:
            self.Q = np.zeros(shape=shape, dtype=float)
        else:
            self.Q = np.zeros((self.inputs.ny, self.inputs.nx, 4), dtype=float)

    def clear_cache(self) -> None:
        self.cache.clear()

    def from_state(self, state: State):
        self._Q = self.converter.convert(state=state)

    @abstractmethod
    def non_dim(self):
        """
        Makes state vector and state variables non-dimensional
        """
        raise NotImplementedError

    @abstractmethod
    def a(self):
        """
        Returns speed of sound over entire grid
        """
        raise NotImplementedError

    @abstractmethod
    def H(self):
        """
        Returns total entalpy over entire grid
        """
        raise NotImplementedError

    def reshape(self, shape: tuple):
        self.Q = self.Q.reshape(shape)
