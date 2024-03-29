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

import pyhype.limiters.limiters as limiters
from pyhype.factory import Factory

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.limiters.base import SlopeLimiter
    from pyhype.solvers.base import SolverConfig


class SlopeLimiterFactory(Factory):
    @classmethod
    def create(cls, config: SolverConfig, **kwargs) -> SlopeLimiter:
        if config.fvm_slope_limiter_type == "VanLeer":
            return limiters.VanLeer(config)
        if config.fvm_slope_limiter_type == "VanAlbada":
            return limiters.VanAlbada(config)
        if config.fvm_slope_limiter_type == "Venkatakrishnan":
            return limiters.Venkatakrishnan(config)
        if config.fvm_slope_limiter_type == "BarthJespersen":
            return limiters.BarthJespersen(config)
        raise ValueError("MUSCL: Slope limiter type not specified.")
