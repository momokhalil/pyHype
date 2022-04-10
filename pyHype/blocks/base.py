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
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import functools
import numpy as np
import matplotlib.pyplot as plt
import pyHype.blocks.QuadBlock as Qb
from pyHype.states import PrimitiveState, ConservativeState

from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock
    from pyHype.solvers.base import ProblemInput


class Neighbors:
    """
    A class that holds references to a Block's neighbors.

    :ivar E: Reference to the east neighbor
    :ivar W: Reference to the west neighbor
    :ivar N: Reference to the north neighbor
    :ivar S: Reference to the south neighbor
    :ivar NE: Reference to the north-east neighbor
    :ivar NW: Reference to the north-west neighbor
    :ivar SE: Reference to the south-east neighbor
    :ivar SW: Reference to the south-west neighbor
    """

    def __init__(self,
                 E: QuadBlock = None,
                 W: QuadBlock = None,
                 N: QuadBlock = None,
                 S: QuadBlock = None,
                 NE: QuadBlock = None,
                 NW: QuadBlock = None,
                 SE: QuadBlock = None,
                 SW: QuadBlock = None
                 ) -> None:
        """

        :type E: QuadBlock
        :param E: East neighbor block

        :type W: QuadBlock
        :param W: West neighbor block

        :type N: QuadBlock
        :param N: North neighbor block

        :type S: QuadBlock
        :param S: South neighbor block

        :type NE: QuadBlock
        :param NE: North-East neighbor block

        :type NW: QuadBlock
        :param NW: North-West neighbor block

        :type SE: QuadBlock
        :param SE: South-East neighbor block

        :type SW: QuadBlock
        :param SW: South-West neighbor block

        :rtype: None
        :return: None
        """
        self.E = E
        self.W = W
        self.N = N
        self.S = S
        self.NE = NE
        self.NW = NW
        self.SE = SE
        self.SW = SW


class NormalVector:
    """
    A class that holds the x- and y-components of a normal vector, calculated based on a given angle theta.

    :ivar x: x-component of the normal vector
    :ivar y: x-component of the normal vector
    """
    def __init__(self,
                 theta: float
                 ) -> None:
        """
        Instantiates the class and calculates the x- and y-components based on the given angle theta.

        :type theta: float
        :param theta: angle in radians

        :rtype: None
        :return: None
        """
        if theta == 0:
            self.x, self.y = 1, 0
        elif theta == np.pi / 2:
            self.x, self.y = 0, 1
        elif theta == np.pi:
            self.x, self.y = -1, 0
        elif theta == 3 * np.pi / 2:
            self.x, self.y = 0, -1
        elif theta == 2 * np.pi:
            self.x, self.y = 1, 0
        else:
            self.x = np.cos(theta)
            self.y = np.sin(theta)

    def __str__(self) -> str:
        """
        Print type, including the values of x and y.

        :rtype: str
        :return: String that decribes the class and provides the values of x and y
        """
        return 'NormalVector object: [' + str(self.x) + ', ' + str(self.y) + ']'


class Blocks:
    def __init__(self,
                 inputs
                 ) -> None:
        # Set inputs
        self.inputs = inputs
        # Number of blocks
        self.num_BLK = None
        # Blocks dictionary
        self.blocks = {}
        # cpu handling this list of blocks
        self.cpu = None

        # Build blocks
        self.build()

    @staticmethod
    def to_all_blocks(func: Callable):
        @functools.wraps(func)
        def _wrapper(self, *args, **kwargs):
            for block in self.blocks.values():
                func(self, block, *args, **kwargs)
        return _wrapper

    def __getitem__(self,
                    blknum: int
                    ) -> QuadBlock:
        return self.blocks[blknum]

    def add(self,
            block: QuadBlock
            ) -> None:
        self.blocks[block.global_nBLK] = block

    def update(self,
               dt: float,
               ) -> None:
        for block in self.blocks.values():
            block.update(dt)

    def set_BC(self) -> None:
        for block in self.blocks.values():
            block.set_BC()

    def build(self) -> None:
        for BLK_data in self.inputs.mesh_inputs.values():
            self.add(Qb.QuadBlock(self.inputs, BLK_data))

        self.num_BLK = len(self.blocks)

        for global_nBLK, block in self.blocks.items():
            Neighbor_E_n = self.inputs.mesh_inputs.get(block.global_nBLK).NeighborE
            Neighbor_W_n = self.inputs.mesh_inputs.get(block.global_nBLK).NeighborW
            Neighbor_N_n = self.inputs.mesh_inputs.get(block.global_nBLK).NeighborN
            Neighbor_S_n = self.inputs.mesh_inputs.get(block.global_nBLK).NeighborS

            block.connect(NeighborE=self.blocks[Neighbor_E_n] if Neighbor_E_n is not None else None,
                          NeighborW=self.blocks[Neighbor_W_n] if Neighbor_W_n is not None else None,
                          NeighborN=self.blocks[Neighbor_N_n] if Neighbor_N_n is not None else None,
                          NeighborS=self.blocks[Neighbor_S_n] if Neighbor_S_n is not None else None,
                          NeighborNE=None,
                          NeighborNW=None,
                          NeighborSE=None,
                          NeighborSW=None)

    def print_connectivity(self) -> None:
        for _, block in self.blocks.items():
            print('-----------------------------------------')
            print('CONNECTIVITY FOR GLOBAL BLOCK: ', block.global_nBLK, '<{}>'.format(block))
            print('North: ', block.neighbors.N)
            print('South: ', block.neighbors.S)
            print('East:  ', block.neighbors.E)
            print('West:  ', block.neighbors.W)

    def plot_mesh(self):
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        for block in self.blocks.values():
            block.plot(ax=ax)
        plt.show()
        plt.pause(0.001)
        plt.close()


class BaseBlock:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    @staticmethod
    def _is_all_blk_conservative(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, ConservativeState), blks))

    @staticmethod
    def _is_all_blk_primitive(blks: dict.values):
        return all(map(lambda blk: isinstance(blk.state, PrimitiveState), blks))


class BaseBlock_Only_State(BaseBlock):
    def __init__(self, inputs: ProblemInput, nx: int, ny: int, state_type: str = 'conservative'):
        super().__init__(inputs)
        if state_type == 'conservative':
            self.state = ConservativeState(inputs, nx=nx, ny=ny)
        elif state_type == 'primitive':
            self.state = PrimitiveState(inputs, nx=nx, ny=ny)
        else:
            raise TypeError('BaseBlock_Only_State.__init__(): Undefined state type.')
