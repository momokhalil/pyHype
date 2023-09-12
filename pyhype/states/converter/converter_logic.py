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
from abc import ABC
from typing import TYPE_CHECKING, Type, Callable

import pyhype.states as states

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np

if TYPE_CHECKING:
    from pyhype.states.base import State


class ConverterLogic(ABC):
    """
    Defines interface for implementing State converters.
    """

    @classmethod
    def get_func(
        cls, state_type: Type[states.State]
    ) -> Callable[[states.State, bool], np.ndarray]:
        """
        Returns the conversion function that converts a Base type to a state_type.

        :param state_type: State type to get conversion function for
        :return:
        """
        if state_type == states.PrimitiveState:
            return cls.to_primitive
        if state_type == states.ConservativeState:
            return cls.to_conservative

    @staticmethod
    def to_primitive(state: states.State, copy: bool = True) -> np.ndarray:
        """
        Defines the conversion logic to convert from the base state type to the
        primitive state type. This shall return a numpy array filled with the new
        state values. The array then gets built into the correct State object type
        inside the StateConverter.

        Defaults to returning the input state array, or its copy.

        :param state: The State object to convert
        :param copy: To copy the state array if converting to the same type
        :return: Numpy array with the correct data values
        """
        return state.data.copy() if copy else state.data

    @staticmethod
    def to_conservative(state: states.State, copy: bool = True) -> np.ndarray:
        """
        Defines the conversion logic to convert from the base state type to the
        conservative state type. This shall return a numpy array filled with the new
        state values. The array then gets built into the correct State object type
        inside the StateConverter.

        Defaults to returning the input state array, or its copy.

        :param state: The State object to convert
        :param copy: To copy the state array if converting to the same type
        :return: Numpy array with the correct data values
        """
        return state.data.copy() if copy else state.data


class ConservativeConverter(ConverterLogic):
    """
    Converts `ConservativeState` objects to other `State` types.
    """

    @staticmethod
    def to_primitive(state: states.ConservativeState, copy: bool = True) -> np.ndarray:
        """
        Logic that converts from a conservative state into a primitive state.

        :param state: ConservativeState object to convert
        :param copy: To copy the state array if converting to the same type
        :return: Numpy array with the equivalent state in the primitive basis
        """
        return np.dstack(
            (
                state.rho.copy(),
                state.u,
                state.v,
                (state.fluid.gamma() - 1) * (state.e - state.ek()),
            )
        )


class PrimitiveConverter(ConverterLogic):
    @staticmethod
    def to_conservative(state: states.PrimitiveState, copy: bool = True) -> np.ndarray:
        """
        Logic that converts from a primitive state into a conservative state.

        :param state: PrimitiveState object to convert
        :param copy: To copy the state array if converting to the same type
        :return: Numpy array with the equivalent state in the primitive basis
        """
        return np.dstack(
            (
                state.rho.copy(),
                state.rho * state.u,
                state.rho * state.v,
                state.p / (state.fluid.gamma() - 1) + state.ek(),
            )
        )
