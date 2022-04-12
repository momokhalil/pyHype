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

import numpy as np
from abc import abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput

class State:
    """
    # State
    Defines an abstract class for implementing primitive and conservative state classes. The core components of a state
    are the state vector and the state variables. The state vector is composed of the state variables in a specific
    order. For example, for a state X with state variables $x_1, x_2, ..., x_n$ and state vector $X$, the state vector
    is represented as:
    $X = \\begin{bmatrix} x_1 \\ x_2 \\ \\dots \\ x_n \\end{bmatrix}^T$. The state vector represents the solution at
    each physical discretization point.
    """

    def __init__(self, inputs: ProblemInput, nx: int, ny: int):
        """
        ## Attributes

        **Private**                                 \n
            input       input dictionary            \n
            size        size of grid in block       \n

        **Public**                                  \n
            g           (gamma) specific heat ratio \n
        """

        # Private
        self.inputs = inputs
        self.nx = nx
        self.ny = ny
        self.g = inputs.gamma
        self._Q = np.zeros((ny, nx, 4), dtype=float)
        self.g_over_gm = self.g / (self.g - 1)
        self.one_over_gm = 1 / (self.g - 1)
        self.cache = {}

    @property
    def Q(self):
        return self._Q

    @Q.setter
    def Q(self, Q):
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

    def clear_cache(self) -> None:
        self.cache.clear()

    def scopy(self):
        return self

    def dcopy(self):
        _copy = State(self.inputs, self.nx, self.ny)
        _copy.Q = self.Q.copy()

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Overload __getitem__ method to return slice from W based on index slice object/indices
        """
        return self.Q[index]

    def __add__(self, other: [State, np.ndarray]) -> np.ndarray:
        """
        Overload __add__ method to return the sum of self and other's state vectors
        """
        if isinstance(other, State):
            return self.Q + other.Q
        if isinstance(other, np.ndarray):
            return self.Q + other

    def __radd__(self, other: [State, np.ndarray]) -> np.ndarray:
        """
        Overload __radd__ method to return the sum of self and other's state vectors
        """
        if isinstance(other, State):
            return other.Q + self.Q
        if isinstance(other, np.ndarray):
            return other + self.Q

    def __sub__(self, other: State) -> np.ndarray:
        """
        Overload __sub__ method to return the difference between self and other's state vectors
        """
        if isinstance(other, State):
            return self.Q - other.Q
        if isinstance(other, np.ndarray):
            return self.Q - other

    def __rsub__(self, other: [State, np.ndarray]) -> np.ndarray:
        """
        Overload __rsub__ method to return the sum of self and other's state vectors
        """
        if isinstance(other, State):
            return other.Q - self.Q
        if isinstance(other, np.ndarray):
            return other - self.Q

    def reset(self, shape: [int] = None):
        if shape:
            self.Q = np.zeros(shape=shape, dtype=float)
        else:
            self.Q = np.zeros((self.ny, self.nx, 4), dtype=float)

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
