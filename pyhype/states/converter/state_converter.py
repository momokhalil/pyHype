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


class StateConverter(ABC):
    def __init__(self):
        self._converter_map = {
            states.PrimitiveState: PrimitiveConverter,
            states.ConservativeState: ConservativeConverter,
        }

    def _get_conversion_func(
        self, from_type: Type[states.State], to_type: Type[states.State]
    ):
        return self._converter_map[from_type].get_func(state_type=to_type)

    def from_state(
        self, state: states.State, from_state: states.State, copy: bool = True
    ) -> None:
        """
        Copies the data from from_state into state, while converting the data's variable
        basis from from_state's type to state's type.

        Example:
        If from_state is a PrimitiveState, and state is a ConservativeState, it will set
        state's internal data to from_state's internal data, while converting it into the
        conservative variable basis.

        :param state: The state to copy data into
        :param from_state: The state to copy data from
        :param copy: To copy the state array if converting to the same type
        :return: None
        """
        if not state.shape == from_state.shape:
            raise ValueError(
                f"States must have equal shape, but state has {state.shape} and from_state has {from_state.shape}"
            )
        func = self._get_conversion_func(
            from_type=type(from_state),
            to_type=type(state),
        )
        state.data = func(state=from_state, copy=copy)

    def to_type(
        self,
        state: states.State,
        to_type: Type[states.State],
        copy: bool = True,
    ) -> states.State:
        """
        Creates a new State from state, with type to_type.

        :param state: The state to create from
        :param to_type: The type of the new state
        :param copy: To copy the state array if converting to the same type
        :return: State with type to_type
        """
        func = self._get_conversion_func(
            from_type=type(state),
            to_type=to_type,
        )
        return to_type(fluid=state.fluid, array=func(state=state, copy=copy))
