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

if TYPE_CHECKING:
    from pyhype.flux.base import FluxFunction
    from pyhype.solvers.base import ProblemInput


class FluxFunctionFactory:
    @classmethod
    def create(cls, inputs: ProblemInput, type: str = "Roe") -> FluxFunction:
        if type == "Roe":
            flux_function_X = FluxRoe(inputs, size=inputs.nx, sweeps=inputs.ny)
            flux_function_Y = FluxRoe(inputs, size=inputs.ny, sweeps=inputs.nx)
            return flux_function_X, flux_function_Y
        if type == "HLLE":
            flux_function_X = FluxHLLE(inputs, nx=inputs.nx, ny=inputs.ny)
            flux_function_Y = FluxHLLE(inputs, nx=inputs.ny, ny=inputs.nx)
            return flux_function_X, flux_function_Y
        if type == "HLLL":
            flux_function_X = FluxHLLL(inputs, nx=inputs.nx, ny=inputs.ny)
            flux_function_Y = FluxHLLL(inputs, nx=inputs.ny, ny=inputs.nx)
            return flux_function_X, flux_function_Y
        raise ValueError(f"Flux function type {type} is not available.")
