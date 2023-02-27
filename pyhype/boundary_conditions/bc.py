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

from typing import TYPE_CHECKING
from pyhype.states import PrimitiveState
from pyhype.boundary_conditions.base import BoundaryCondition

if TYPE_CHECKING:
    from pyhype.states import State


class PrimitiveDirichletBC(BoundaryCondition):
    def __init__(self, primitive_state: PrimitiveState):
        if not isinstance(primitive_state, PrimitiveState):
            raise TypeError("primitive_array must be a PrimitiveState.")
        super().__init__()
        self._primitive_state = primitive_state
        self._primitive_state.make_non_dimensional()

    def _apply_boundary_condition(self, state: State, *args, **kwargs) -> None:
        state.from_state(self._primitive_state)
