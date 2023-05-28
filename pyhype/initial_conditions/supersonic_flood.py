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
from typing import TYPE_CHECKING

from pyhype.states.primitive import PrimitiveState
from pyhype.states.conservative import ConservativeState
from pyhype.initial_conditions.base import InitialCondition

if TYPE_CHECKING:
    from pyhype.fluids.base import Fluid
    from pyhype.blocks.quad_block import QuadBlock

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class SupersonicFloodInitialCondition(InitialCondition):
    def __init__(self, fluid: Fluid, rho: float, u: float, v: float, p: float):
        if rho <= 0 or p <= 0:
            raise ValueError(f"Unrealizable density (rho={rho}) or pressure (p={p}).")
        self._rho = rho
        self._u = u
        self._v = v
        self._p = p

        a = np.sqrt(fluid.gamma() * p / rho)
        velocity = np.hypot(u, v)
        mach_number = velocity / a

        if mach_number < 1.0:
            raise ValueError(
                "The given set of conditions do not produce a supersonic flow:\n"
                f"Speed of Sound = {a}, Total Velocity = {velocity}, Mach Number = {mach_number}."
            )

    def apply_to_block(self, block: QuadBlock):
        state = PrimitiveState(
            fluid=block.config.fluid,
            array=np.array([self._rho, self._u, self._v, self._p]).reshape((1, 1, 4)),
        ).to_type(ConservativeState)

        block.state.data = state.data
        block.state.make_non_dimensional()
