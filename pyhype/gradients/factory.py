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
from pyhype.factory import Factory
from pyhype.gradients.greengauss import GreenGauss

if TYPE_CHECKING:
    from pyhype.solvers.base import ProblemInput


class GradientFactory(Factory):
    @classmethod
    def create(cls, type: str, inputs: ProblemInput, **kwargs):
        """
        Creates a concrete object of type SolverComponent.

        :type type: str
        :param type: Type of object to be created
        """
        if type == "GreenGauss":
            return GreenGauss(inputs=inputs)
        raise ValueError(f"Gradient type {type} is not available.")
