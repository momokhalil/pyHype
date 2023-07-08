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

from abc import ABC
from typing import TYPE_CHECKING, Type

import pyhype.states as states
from pyhype.states.converter.concrete_defs import (
    PrimitiveConverter,
    ConservativeConverter,
)

if TYPE_CHECKING:
    from pyhype.states.base import State
    from pyhype.states.converter.concrete_defs import ConverterLogic


class StateConverter(ABC):
    @staticmethod
    def get_converter(state_type: Type[states.State]) -> Type[ConverterLogic]:
        """
        Returns the converter type associated with state_type

        :param state_type: Type of state to get converter for
        :return:
        """
        if state_type == states.PrimitiveState:
            return PrimitiveConverter
        if state_type == states.ConservativeState:
            return ConservativeConverter

    def from_state(self, state: states.State, from_state: states.State) -> None:
        """
        Copies the data from from_state into state, while converting the data's variable
        basis from from_state's type to state's type.

        Example:
        If from_state is a PrimitiveState, and state is a ConservativeState, it will set
        state's internal data to from_state's internal data, while converting it into the
        conservative variable basis.

        :param state: The state to copy data into
        :param from_state: The state to copy data from
        :return: None
        """
        if not state.shape == from_state.shape:
            raise ValueError(
                f"States must have equal shape, but state has {state.shape} and from_state has {from_state.shape}"
            )
        converter = self.get_converter(type(from_state))
        func = converter.get_func(state_type=type(state))
        state.data = func(state=from_state)

    def to_type(
        self,
        state: states.State,
        to_type: Type[states.State],
    ) -> states.State:
        """
        Creates a new State from state, with type to_type.

        :param state: The state to create from
        :param to_type: The type of the new state
        :return: State with type to_type
        """
        converter = self.get_converter(type(state))
        func = converter.get_func(state_type=to_type)
        array = func(state=state)
        created = to_type(fluid=state.fluid, array=array)
        return created
