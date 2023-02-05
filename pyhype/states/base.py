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

from typing import Union, Type
from abc import ABC, abstractmethod
from pyhype.fluids.base import Fluid
from pyhype.states.converter import StateConverter, BaseConverter

import numpy as np

StateCompatible = (int, float, np.ndarray)


class RealizabilityException(Exception):
    pass


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

    def __init__(
        self,
        fluid: Fluid,
        state: State = None,
        array: np.ndarray = None,
        shape: tuple[int, int, int] = None,
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

        if not isinstance(fluid, Fluid):
            raise TypeError("fluid must be of type Fluid.")

        self.fluid = fluid
        self.converter = StateConverter()
        self._data = None
        self.cache = {}

        if state is not None:
            self._data = np.empty(state.shape)
            self.from_state(state)
        elif array is not None:
            self.from_array(array)
        elif shape is not None:
            self.data = np.full(
                shape=shape, fill_value=fill if fill is not None else 0.0
            )
        else:
            raise ValueError(
                "State constructor must recieve either a state, array or a shape."
            )

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        self.from_array(data)

    def make_non_dimensional(self):
        self.data[:, :, 0] /= self.fluid.far_field.rho
        self.data[:, :, 1] /= self.fluid.far_field.rho * self.fluid.far_field.a
        self.data[:, :, 2] /= self.fluid.far_field.rho * self.fluid.far_field.a
        self.data[:, :, 3] /= self.fluid.far_field.rho * self.fluid.far_field.a**2

    def _set_data_array_from_array(self, array: np.ndarray):
        if self.data is None:
            self._data = array
        elif self.shape == array.shape:
            self._data = array
        else:
            # Broadcast to array, if this fails then
            # data has an incompatible shape
            self._data[:, :, :] = array

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

    @property
    def shape(self):
        return self._data.shape

    @property
    def nx(self):
        return self.shape[1]

    def ny(self):
        return self.shape[0]

    def __getitem__(self, index: Union[int, slice]) -> State:
        return type(self)(fluid=self.fluid, array=self.data[index].copy())

    def __add__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=self.data + other.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=self.data + other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __radd__(self, other: Union[State, StateCompatible]) -> State:
        return self.__add__(other)

    def __sub__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=self.data - other.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=self.data - other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rsub__(self, other: Union[State, StateCompatible]) -> State:
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=other.data - self.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=other - self.data)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __mul__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=self.data * other.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=self.data * other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rmul__(self, other: Union[State, StateCompatible]):
        return self.__mul__(other)

    def __truediv__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=self.data / other.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=self.data / other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rtruediv__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=other.data / self.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=other / self.data)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __pow__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=self.data**other.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=self.data**other)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def __rpow__(self, other: Union[State, StateCompatible]):
        if isinstance(other, type(self)):
            return type(self)(fluid=self.fluid, array=other.data**self.data)
        if isinstance(other, StateCompatible):
            return type(self)(fluid=self.fluid, array=other**self.data)
        raise TypeError(
            f"Other must be of type {self.__class__.__name__} or {StateCompatible}"
        )

    def reset(self, shape: tuple[int] = None):
        if shape:
            self.data = np.zeros(shape=shape, dtype=float)
        else:
            self.data = np.zeros((self.ny, self.nx, 4), dtype=float)

    def clear_cache(self) -> None:
        self.cache.clear()

    def from_state(self, state: State):
        self.converter.from_state(state=self, from_state=state)
        self.clear_cache()

    def to_type(self, to_type: Type[State]):
        return self.converter.to_type(state=self, to_type=to_type)

    def from_array(self, array: np.ndarray):
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"Input array must be a Numpy array, but it is a {type(array)}."
            )
        if array.ndim != 3 or array.shape[-1] != 4:
            raise ValueError("Array must have 3 dims and a depth of 4.")
        self._set_data_array_from_array(array)
        self.clear_cache()

    def transpose(self, axes: [int]):
        self._data = self._data.transpose(axes)

    def realizable(self):
        """
        Checks if a State is realizable.
        :raises: RealizabilityException
        """
        conditions = self.realizability_conditions()
        if all(np.all(condition) for condition in conditions.values()):
            return True
        bad_values = {
            name: np.where(np.bitwise_not(good_vals))
            for name, good_vals in conditions.items()
            if not np.all(good_vals)
        }
        print("ConservativeState has bad values in the following conditions:")
        print("-------------------------------------------------------------")
        for condition_name, bad_val_indices in bad_values.items():
            print(f"Condition: {condition_name}, location of bad values:")
            print(bad_val_indices)
        raise RealizabilityException(
            "Simulation has failed due to an non-realizable state quantity."
        )

    @abstractmethod
    def realizability_conditions(self) -> dict[str, np.ndarray]:
        """
        Returns the conditions that must be True in every cell for the
        State to be realizable.
        :return: conditions dict
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

    @abstractmethod
    def get_class_type_converter(self) -> Type[BaseConverter]:
        """
        Returns the converter class that converts objects of this class
        :return:
        """
        raise NotImplementedError

    def reshape(self, shape: tuple):
        self._data = np.reshape(self._data, shape)
