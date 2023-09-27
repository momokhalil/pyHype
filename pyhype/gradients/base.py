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
    from pyhype.solvers.base import SolverConfig
    from pyhype.blocks.quad_block import BaseBlockGhost


class Gradient:
    def __init__(self, config: SolverConfig):
        self.config = config

    def compute(self, parent_block: BaseBlockGhost) -> None:
        """
        Interface to call the gradient algorithm.

        :type parent_block: BaseBlockGhost
        :param parent_block: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        self._get_gradient(parent_block)

    @abstractmethod
    def _get_gradient(self, parent_block: BaseBlockGhost) -> None:
        """
        Implementation of the gradient algorithm.

        :type parent_block: BaseBlockGhost
        :param parent_block: Solution block containing state solution and mesh geometry data

        :rtype: None
        :return: None
        """
        raise NotImplementedError


"""
class LeastSquares9Point:
    def __init__(self, config: SolverConfig):
        self.config = config

        self.stencilSW = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]]
        self.stencilNW = [[-2, 0], [-1, 0], [0, 0], [0, 1], [-2, 0], [-2, 1], [-1, 1]]
        self.stencilSE = [[0, 0], [1, 0], [0, -1], [0, -2], [0, -1], [1, -1], [1, -2]]
        self.stencilNE = [[0, -1], [0, -2], [-1, 0], [-2, 0], [-1, -2], [1, -1], [1, -2]]

    def __call__(self, parent_block):
        return self.least_squares_nearest_neighbor(parent_block)

    def least_squares_nearest_neighbor(self, parent_block: BaseBlockGhost):
        bdr = parent_block.boundary_blocks

        parent_block.grad.x, parent_block.grad.y = least_squares_9_point(parent_block.state.data,
                                           bdr.E.state.data, bdr.W.state.data, bdr.N.state.data, bdr.S.state.data,
                                           parent_block.mesh.x, parent_block.mesh.y,
                                           bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                           bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                           self.config.nx, self.config.ny,
                                           self.stencilSW, self.stencilNW, self.stencilSE, self.stencilNE)
"""
