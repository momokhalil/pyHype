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

from pyhype.states.conservative import ConservativeState

from typing import Type, TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.initial_conditions.base import InitialCondition
    from pyhype.fluids.base import Fluid
    from pyhype.states.base import State


class SolverConfig:
    __slots__ = [
        "fvm_type",
        "fvm_spatial_order",
        "fvm_num_quadrature_points",
        "fvm_gradient_type",
        "fvm_flux_function_type",
        "fvm_slope_limiter_type",
        "time_integrator",
        "initial_condition",
        "interface_interpolation",
        "reconstruction_type",
        "write_solution",
        "write_solution_mode",
        "write_solution_name",
        "write_every_n_timesteps",
        "plot_every",
        "plot_function",
        "CFL",
        "t_final",
        "realplot",
        "profile",
        "fluid",
        "nx",
        "ny",
        "n",
        "nghost",
        "use_JIT",
    ]

    def __init__(
        self,
        nx: int,
        ny: int,
        CFL: float,
        t_final: float,
        initial_condition: InitialCondition,
        fvm_type: str,
        time_integrator: str,
        fvm_gradient_type: str,
        fvm_flux_function_type: str,
        fvm_slope_limiter_type: str,
        fvm_spatial_order: int,
        fvm_num_quadrature_points: int,
        fluid: Fluid,
        nghost: int = 1,
        use_JIT: bool = True,
        profile: bool = False,
        realplot: bool = False,
        plot_every: int = 20,
        plot_function: str = "Density",
        write_solution: bool = False,
        write_solution_mode: str = "every_n_timesteps",
        write_solution_name: str = "nozzle",
        reconstruction_type: Type[State] = ConservativeState,
        write_every_n_timesteps: int = 40,
        interface_interpolation: str = "arithmetic_average",
    ) -> None:

        self.initial_condition = initial_condition

        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.nghost = nghost

        self.CFL = CFL
        self.t_final = t_final

        self.fvm_type = fvm_type
        self.time_integrator = time_integrator
        self.fvm_gradient_type = fvm_gradient_type
        self.fvm_flux_function_type = fvm_flux_function_type
        self.fvm_slope_limiter_type = fvm_slope_limiter_type
        self.fvm_spatial_order = fvm_spatial_order
        self.fvm_num_quadrature_points = fvm_num_quadrature_points

        self.reconstruction_type = reconstruction_type
        self.interface_interpolation = interface_interpolation

        self.fluid = fluid

        self.use_JIT = use_JIT
        self.profile = profile
        self.realplot = realplot
        self.plot_every = plot_every
        self.plot_function = plot_function

        self.write_solution = write_solution
        self.write_solution_mode = write_solution_mode
        self.write_solution_name = write_solution_name
        self.write_every_n_timesteps = write_every_n_timesteps

    def __str__(self):
        return "".join(f"\t{atr}: {getattr(self, atr)}\n" for atr in self.__slots__)
