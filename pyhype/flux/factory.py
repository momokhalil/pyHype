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

from .Roe import FluxRoe
from .HLLE import FluxHLLE
from .HLLL import FluxHLLL

from typing import TYPE_CHECKING
from functools import partial
from pyhype.factory import Factory

if TYPE_CHECKING:
    from pyhype.flux.base import FluxFunction
    from pyhype.solvers.base import ProblemInput


class FluxFunctionFactory(Factory):
    @classmethod
    def create(cls, inputs: ProblemInput, type: str = "Roe", **kwargs) -> FluxFunction:
        if type == "Roe":
            flux_func_x = FluxRoe(inputs, size=inputs.nx, sweeps=inputs.ny)
            flux_func_y = FluxRoe(inputs, size=inputs.ny, sweeps=inputs.nx)
            return flux_func_x, flux_func_y
        if type == "HLLE":
            flux_func_x = FluxHLLE(inputs, nx=inputs.nx, ny=inputs.ny)
            flux_func_y = FluxHLLE(inputs, nx=inputs.ny, ny=inputs.nx)
            return flux_func_x, flux_func_y
        if type == "HLLL":
            flux_func_x = FluxHLLL(inputs, nx=inputs.nx, ny=inputs.ny)
            flux_func_y = FluxHLLL(inputs, nx=inputs.ny, ny=inputs.nx)
            return flux_func_x, flux_func_y
        raise ValueError(f"Flux function type {type} is not available.")
