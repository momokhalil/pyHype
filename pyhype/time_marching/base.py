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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.solver_config import SolverConfig
    from pyhype.blocks.base import Blocks


class TimeIntegrator:
    def __init__(self, config: SolverConfig):
        self.config = config

    def __call__(self, dt: float, blocks: Blocks):
        self.integrate(dt, blocks)

    @abstractmethod
    def integrate(self, dt: float, blocks: Blocks):
        raise NotImplementedError
