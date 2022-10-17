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

import numpy as np
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from pyhype.states import PrimitiveState

if TYPE_CHECKING:
    from pyhype.states import State


class BoundaryCondition(ABC):
    def __call__(self, state: State, *args, **kwargs) -> None:
        self._apply_boundary_condition(state, *args, **kwargs)

    @abstractmethod
    def _apply_boundary_condition(self, state: State, *args, **kwargs) -> None:
        raise NotImplementedError


class PrimitiveDirichletBC(BoundaryCondition):
    def __init__(self, primitive_array: np.ndarray):
        if not isinstance(primitive_array, np.ndarray):
            raise TypeError("primitive_array must be a numpy array.")
        if primitive_array.ndim != 3 and primitive_array.shape[-1] != 4:
            raise ValueError("primitive array has an incompatible shape.")
        super().__init__()
        self._primitive_array = primitive_array

    def _apply_boundary_condition(self, state: State, *args, **kwargs) -> None:
        inlet_state = PrimitiveState(inputs=state.inputs, array=self._primitive_array)
        state.from_state(inlet_state)
