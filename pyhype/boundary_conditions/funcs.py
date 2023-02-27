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
from typing import Union, TYPE_CHECKING
import pyhype.utils.utils as utils

if TYPE_CHECKING:
    from pyhype.states import State


class BoundaryConditionFunctions:
    @staticmethod
    def reflection(state: State, wall_angle: Union[np.ndarray, int, float]) -> None:
        """
        Flips the sign of the u velocity along the wall. Rotates the state
        from global to wall frame and back to ensure
        coordinate alignment.

        Parameters:
            - state (np.ndarray): Ghost cell state arrays
            - wall_angle (np.ndarray): Array of wall angles at each point along the wall

        Returns:
            - None
        """
        utils.rotate(wall_angle, state.data)
        state.data[:, :, 1] = -state.data[:, :, 1]
        utils.unrotate(wall_angle, state.data)
