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
from typing import TYPE_CHECKING

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np

from pyHype.blocks.ghost import GhostBlock

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock


class Neighbors:
    def __init__(self,
                 E: QuadBlock = None,
                 W: QuadBlock = None,
                 N: QuadBlock = None,
                 S: QuadBlock = None
                 ) -> None:
        """
        A class designed to hold references to each Block's neighbors.

        Parameters:
            - E: Reference to the east neighbor
            - W: Reference to the west neighbor
            - N: Reference to the north neighbor
            - S: Reference to the south neighbor
        """

        self.E = E
        self.W = W
        self.N = N
        self.S = S


class GhostBlockContainer:
    def __init__(self,
                 E: GhostBlock = None,
                 W: GhostBlock = None,
                 N: GhostBlock = None,
                 S: GhostBlock = None
                 ) -> None:
        """
        A class designed to hold references to each Block's ghost blocks.

        Parameters:
            - E: Reference to the east ghost block
            - W: Reference to the west nghost block
            - N: Reference to the north ghost block
            - S: Reference to the south ghost block
        """

        self.E = E
        self.W = W
        self.N = N
        self.S = S


class NormalVector:
    def __init__(self,
                 theta: float
                 ) -> None:

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

    def __str__(self):
        return 'NormalVector object: [' + str(self.x) + ', ' + str(self.y) + ']'


class Blocks:
    def __init__(self,
                 inputs
                 ) -> None:

        self.inputs = inputs
        self.number_of_blocks = None
        self.blocks = {}

        self.build()

    def __getitem__(self,
                    blknum: int
                    ) -> QuadBlock:

        return self.blocks[blknum]

    def add(self,
            block: QuadBlock
            ) -> None:

        self.blocks[block.global_nBLK] = block

    def get(self,
            block: int
            ) -> QuadBlock:

        return self.blocks[block]

    def update(self,
               dt: float
               ) -> None:

        for block in self.blocks.values():
            block.update(dt)

    def set_BC(self) -> None:
        for block in self.blocks.values():
            block.set_BC()

    def update_BC(self) -> None:
        for block in self.blocks.values():
            block.update_BC()

    def build(self) -> None:
        mesh_inputs = self.inputs.mesh_inputs

        for BLK_data in mesh_inputs.values():
            self.add(QuadBlock(self.inputs, BLK_data))

        self.number_of_blocks = len(self.blocks)

        for global_nBLK, block in self.blocks.items():
            Neighbor_E_idx = mesh_inputs.get(block.global_nBLK).NeighborE
            Neighbor_W_idx = mesh_inputs.get(block.global_nBLK).NeighborW
            Neighbor_N_idx = mesh_inputs.get(block.global_nBLK).NeighborN
            Neighbor_S_idx = mesh_inputs.get(block.global_nBLK).NeighborS

            block.connect(NeighborE=self.blocks[Neighbor_E_idx] if Neighbor_E_idx != 0 else None,
                          NeighborW=self.blocks[Neighbor_W_idx] if Neighbor_W_idx != 0 else None,
                          NeighborN=self.blocks[Neighbor_N_idx] if Neighbor_N_idx != 0 else None,
                          NeighborS=self.blocks[Neighbor_S_idx] if Neighbor_S_idx != 0 else None)

    def print_connectivity(self) -> None:
        for _, block in self.blocks.items():
            print('-----------------------------------------')
            print('CONNECTIVITY FOR GLOBAL BLOCK: ', block.global_nBLK, '<{}>'.format(block))
            print('North: ', block.neighbors.N)
            print('South: ', block.neighbors.S)
            print('East:  ', block.neighbors.E)
            print('West:  ', block.neighbors.W)
