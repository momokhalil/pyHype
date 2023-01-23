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

from pyhype.factory import Factory
from pyhype.time_marching import (
    ExplicitRungeKutta as Erk,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.solvers.base import SolverConfig


class TimeIntegratorFactory(Factory):
    @classmethod
    def create(cls, config: SolverConfig, **kwargs):
        """
        Creates a concrete object of type SolverComponent.

        :type config: SolverConfig
        :param config: Solver configuration that contains all user-defined params
        """
        if config.time_integrator == "ExplicitEuler1":
            return Erk.ExplicitEuler1(config)
        if config.time_integrator == "RK2":
            return Erk.RK2(config)
        if config.time_integrator == "Generic2":
            return Erk.Generic2(config)
        if config.time_integrator == "Ralston2":
            return Erk.Ralston2(config)
        if config.time_integrator == "Generic3":
            return Erk.Generic3(config)
        if config.time_integrator == "RK3":
            return Erk.RK3(config)
        if config.time_integrator == "RK3SSP":
            return Erk.RK3SSP(config)
        if config.time_integrator == "Ralston3":
            return Erk.Ralston3(config)
        if config.time_integrator == "RK4":
            return Erk.RK4(config)
        if config.time_integrator == "Ralston4":
            return Erk.Ralston4(config)
        if config.time_integrator == "DormandPrince5":
            return Erk.DormandPrince5(config)
        raise ValueError(
            f"Time marching scheme {config.time_integrator} is not available."
        )
