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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type, Callable

import pyhype.states as states

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np

if TYPE_CHECKING:
    from pyhype.states.base import State


class StateConverter(ABC):
    @staticmethod
    def from_state(state: states.State, from_state: states.State) -> None:
        converter = from_state.get_class_type_converter()
        state.data = converter.to(state_type=type(state))(from_state)

    @staticmethod
    def to_type(
        state: states.State,
        to_type: Type[states.State],
    ) -> states.State:
        converter = state.get_class_type_converter()
        array = converter.to(to_type)(state)
        created = to_type(fluid=state.fluid, array=array)
        return created


class BaseConverter(ABC):
    """
    Defines interface for implementing State converters.
    """

    @classmethod
    def to(cls, state_type: Type[states.State]) -> Callable[[states.State], np.ndarray]:
        """
        Returns the conversion function that converts a Base type
        to a state_type.

        :param state_type:
        :return:
        """
        if state_type == states.PrimitiveState:
            return cls.to_primitive
        if state_type == states.ConservativeState:
            return cls.to_conservative

    @staticmethod
    @abstractmethod
    def to_primitive(state: states.State) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def to_conservative(state: states.State) -> np.ndarray:
        raise NotImplementedError


class ConservativeConverter(BaseConverter):
    """
    Converts `ConservativeState` objects to other `State` types.
    """

    @staticmethod
    def to_primitive(state: states.ConservativeState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.u.copy(),
                state.v.copy(),
                (state.fluid.gamma() - 1) * (state.e - state.ek()),
            )
        )

    @staticmethod
    def to_conservative(state: states.ConservativeState) -> np.ndarray:
        return state.data.copy()


class PrimitiveConverter(BaseConverter):
    @staticmethod
    def to_primitive(state: states.PrimitiveState):
        return state.data.copy()

    @staticmethod
    def to_conservative(state: states.PrimitiveState) -> np.ndarray:
        return np.dstack(
            (
                state.rho.copy(),
                state.rho * state.u,
                state.rho * state.v,
                state.p / (state.fluid.gamma() - 1) + state.ek(),
            )
        )
