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

from .FirstOrderMUSCL import FirstOrderMUSCL
from .SecondOrderMUSCL import SecondOrderMUSCL

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.factory import Factory
    from pyhype.fvm.base import FiniteVolumeMethod
    from pyhype.solvers.base import SolverConfig


class FiniteVolumeMethodFactory:
    @classmethod
    def create(
        cls,
        type: str,
        order: int,
        config: SolverConfig,
        flux: Factory.create,
        limiter: Factory.create,
        gradient: Factory.create,
    ) -> FiniteVolumeMethod:
        if type == "MUSCL":
            if order == 1:
                return FirstOrderMUSCL(
                    config=config, limiter=limiter, flux=flux, gradient=gradient
                )
            if order == 2:
                return SecondOrderMUSCL(
                    config=config, limiter=limiter, flux=flux, gradient=gradient
                )
            raise ValueError(
                f"No MUSCL finite volume method has been specialized with order {order}"
            )
        raise ValueError(
            f"Specified finite volume method type {type} has not been specialized."
        )
