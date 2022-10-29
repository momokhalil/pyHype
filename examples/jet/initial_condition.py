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
    from pyhype.blocks.quad_block import QuadBlock

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class SubsonicJetInitialCondition(InitialCondition):
    @staticmethod
    def apply_to_block(block: QuadBlock):
        rho = 1.0
        p = 1 / block.state.fluid.gamma()
        u = 0.0
        v = 0.0
        state = PrimitiveState(
            fluid=block.inputs.fluid,
            array=np.array([rho, u, v, p]).reshape((1, 1, 4)),
        ).to_type(ConservativeState)

        block.state.data = state.data
        block.state.make_non_dimensional()
