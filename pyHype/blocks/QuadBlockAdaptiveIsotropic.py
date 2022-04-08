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

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pyHype.blocks import QuadBlock
from pyHype.fvm import SecondOrderMUSCL
from pyHype.mesh.base import BlockDescription, Mesh
from pyHype.states.states import ConservativeState, PrimitiveState
from pyHype.blocks.base import NormalVector, GhostBlockContainer, Neighbors
from pyHype.solvers.time_integration.explicit_runge_kutta import ExplicitRungeKutta as Erk
from pyHype.blocks.ghost import GhostBlockEast, GhostBlockWest, GhostBlockSouth, GhostBlockNorth

from copy import deepcopy
from copy import copy as cpy

from itertools import chain


if TYPE_CHECKING:
    from pyHype.solvers.base import ProblemInput


class QuadBlockAdaptiveIsotropic(QuadBlock):
    pass
