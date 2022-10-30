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
from collections import namedtuple
from abc import ABC, abstractmethod

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class Fluid(ABC):

    gas_constant = 8314.0

    _molecular_mass = -np.inf

    def __init__(
        self,
        a_inf: float = 1.0,
        rho_inf: float = 1.0,
    ):
        far_field_type = namedtuple("far_field", ["a", "rho"])
        self._far_field = far_field_type(a_inf, rho_inf)
        self._R = self.gas_constant / self.molecular_mass

    @property
    def far_field(self):
        return self._far_field

    @property
    def R(self):
        return self._R

    @property
    def molecular_mass(self):
        return self._molecular_mass

    @abstractmethod
    def gamma(self, *args, temperature: float = None, **kwargs) -> float:
        raise NotImplementedError

    def g_over_gm1(self, *args, temperature: float = None, **kwargs) -> float:
        g = self.gamma(*args, temperature, **kwargs)
        return g / (g - 1.0)

    def one_over_gm1(self, *args, temperature: float = None, **kwargs) -> float:
        g = self.gamma(*args, temperature, **kwargs)
        return 1.0 / (g - 1.0)
