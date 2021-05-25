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

import numpy as np
from abc import abstractmethod
from pyHype.input.input_file_builder import ProblemInput


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

        # Public
        self.g = inputs.gamma

        # State matrix
        self.Q = np.zeros((ny, nx, 4))

        # State variables
        self.q0 = np.zeros((ny, nx, 1))
        self.q1 = np.zeros((ny, nx, 1))
        self.q2 = np.zeros((ny, nx, 1))
        self.q3 = np.zeros((ny, nx, 1))


    def set_vars_from_state(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.q0     = self.Q[:, :, 0]
        self.q1     = self.Q[:, :, 1]
        self.q2     = self.Q[:, :, 2]
        self.q3     = self.Q[:, :, 3]

    def set_state_from_vars(self):
        """
        Sets primitive variables from primitive state vector
        """
        self.Q[:, :, 0] = self.q0
        self.Q[:, :, 1] = self.q1
        self.Q[:, :, 2] = self.q2
        self.Q[:, :, 3] = self.q3

    # ------------------------------------------------------------------------------------------------------------------
    # Overload magic functions

    # Overload __getitem__ method to return slice from W based on index slice object/indices
    def __getitem__(self, index: int) -> np.ndarray:
        return self.Q[index]

    # Overload __add__ method to return the sum of self and other's state vectors
    def __add__(self, other: 'State') -> np.ndarray:
        return self.Q + other.Q

    # Overload __sub__ method to return the difference between self and other's state vectors
    def __sub__(self, other: 'State') -> np.ndarray:
        return self.Q - other.Q

    def reset(self, shape: tuple[int] = None):

        if shape:
            self.Q = np.zeros(shape=shape)
        else:
            self.Q = np.zeros((self.ny, self.nx, 4))

        self.set_vars_from_state()

    def update(self, value: np.ndarray) -> None:
        self.Q = value
        self.set_vars_from_state()

    @abstractmethod
    def non_dim(self):
        """
        Makes state vector and state variables non-dimensional
        """
        pass

    @abstractmethod
    def a(self):
        """
        Returns speed of sound over entire grid
        """
        pass

    @abstractmethod
    def H(self):
        """
        Returns total entalpy over entire grid
        """
        pass

    @abstractmethod
    def rho(self):
        """
        Returns density over entire grid
        """
        pass

    @abstractmethod
    def u(self):
        """
        Returns x-direction velocity over entire grid
        """
        pass

    @abstractmethod
    def v(self):
        """
        Returns y-direction velocity over entire grid
        """
        pass

    @abstractmethod
    def p(self):
        """
        Returns pressure velocity over entire grid
        """
        pass
