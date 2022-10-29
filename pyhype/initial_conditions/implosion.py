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


class ImplosionInitialCondition(InitialCondition):
    @staticmethod
    def apply_to_block(block: QuadBlock):
        # Free stream
        rhoL = 4.6968
        pL = 404400.0
        uL = 0.0
        vL = 0.0
        left_state = PrimitiveState(
            fluid=block.inputs.fluid,
            array=np.array([rhoL, uL, vL, pL]).reshape((1, 1, 4)),
        ).to_type(ConservativeState)

        # Post shock
        rhoR = 1.1742
        pR = 101100.0
        uR = 0.0
        vR = 0.0
        right_state = PrimitiveState(
            fluid=block.inputs.fluid,
            array=np.array([rhoR, uR, vR, pR]).reshape((1, 1, 4)),
        ).to_type(ConservativeState)

        # Fill state vector in each block
        block.state.data = np.where(
            np.logical_and(block.mesh.x <= 5, block.mesh.y <= 5),
            right_state.data,
            left_state.data,
        )
        block.state.make_non_dimensional()
