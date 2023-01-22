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

from abc import ABC
from functools import partial

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.solvers.base import SolverConfig


class Factory(ABC):
    @classmethod
    def create(cls, config: SolverConfig, **kwargs):
        """
        Creates a concrete object of type SolverComponent.

        :type type: str
        :param type: Type of object to be created
        """
        raise NotImplementedError("Calling a base class create()")

    @classmethod
    def get(cls, config: SolverConfig, **kwargs) -> Factory.create:
        """
        Returns a partially initialized Factory.create method with the passed in kwargs.

        :return: Partially initialized creator function to pass into classes that
        need to create objects of this class
        """
        return partial(cls.create, config, **kwargs)
