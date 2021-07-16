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

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pyHype.mesh.base import BlockDescription, Mesh

from pyHype.fvm import SecondOrderPWL

from pyHype.states.states import ConservativeState
from pyHype.solvers.time_integration.explicit_runge_kutta import ExplicitRungeKutta as Erk

from pyHype.blocks.ghost import GhostBlock,                         \
                                GhostBlockEast, GhostBlockWest,     \
                                GhostBlockSouth, GhostBlockNorth

if TYPE_CHECKING:
    from pyHype.solvers.solver import ProblemInput


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


# QuadBlock Class Definition
class QuadBlock:
    def __init__(self,
                 inputs: ProblemInput,
                 block_data: BlockDescription
                 ) -> None:

        self.inputs             = inputs
        self.mesh               = Mesh(inputs, block_data)
        self.state              = ConservativeState(inputs, nx=inputs.nx, ny=inputs.ny)
        self.global_nBLK        = block_data.nBLK
        self.ghost              = None
        self.neighbors          = None
        self.nx                 = inputs.nx
        self.ny                 = inputs.ny


        # Store vertices for brevity
        vert = self.mesh.vertices

        # Side lengths
        self.LE                 = self._get_side_length(vert.SE, vert.NE)
        self.LW                 = self._get_side_length(vert.SW, vert.NW)
        self.LS                 = self._get_side_length(vert.SE, vert.SW)
        self.LS                 = self._get_side_length(vert.NW, vert.NE)

        # Side angles
        self.thetaE             = self._get_side_angle(vert.SE, vert.NE)
        self.thetaW             = self._get_side_angle(vert.SW, vert.NW) + np.pi
        self.thetaS             = self._get_side_angle(vert.SW, vert.SE) + np.pi
        self.thetaN             = self._get_side_angle(vert.NE, vert.NW)

        # Self normal vectors
        self.nE                 = NormalVector(self.thetaE)
        self.nW                 = NormalVector(self.thetaW)
        self.nS                 = NormalVector(self.thetaS)
        self.nN                 = NormalVector(self.thetaN)

        # Gradient
        self.gradx              = np.zeros_like(self.mesh.x)
        self.grady              = np.zeros_like(self.mesh.y)

        # Set finite volume method
        fvm = self.inputs.fvm

        if fvm == 'SecondOrderPWL':
            self.fvm = SecondOrderPWL(self.inputs, self.global_nBLK)
        else:
            raise ValueError('Specified finite volume method has not been specialized.')

        # Set time integrator
        time_integrator = self.inputs.integrator

        if time_integrator      == 'ExplicitEuler1':
            self._time_integrator = Erk.ExplicitEuler1(self.inputs)
        elif time_integrator    == 'RK2':
            self._time_integrator = Erk.RK2(self.inputs)
        elif time_integrator    == 'Generic2':
            self._time_integrator = Erk.Generic2(self.inputs)
        elif time_integrator    == 'Ralston2':
            self._time_integrator = Erk.Ralston2(self.inputs)
        elif time_integrator    == 'Generic3':
            self._time_integrator = Erk.Generic3(self.inputs)
        elif time_integrator    == 'RK3':
            self._time_integrator = Erk.RK3(self.inputs)
        elif time_integrator    == 'RK3SSP':
            self._time_integrator = Erk.RK3SSP(self.inputs)
        elif time_integrator    == 'Ralston3':
            self._time_integrator = Erk.Ralston3(self.inputs)
        elif time_integrator    == 'RK4':
            self._time_integrator = Erk.RK4(self.inputs)
        elif time_integrator    == 'Ralston4':
            self._time_integrator = Erk.Ralston4(self.inputs)
        elif time_integrator    == 'DormandPrince5':
            self._time_integrator = Erk.DormandPrince5(self.inputs)
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # Build boundary blocks
        self.ghost = GhostBlockContainer(E=GhostBlockEast(self.inputs, BCtype=block_data.BCTypeE, refBLK=self),
                                         W=GhostBlockWest(self.inputs, BCtype=block_data.BCTypeW, refBLK=self),
                                         N=GhostBlockNorth(self.inputs, BCtype=block_data.BCTypeN, refBLK=self),
                                         S=GhostBlockSouth(self.inputs, BCtype=block_data.BCTypeS, refBLK=self))

        #self.plot()


    def plot(self):
        plt.scatter(self.mesh.nodes.x[:, :, 0], self.mesh.nodes.y[:, :, 0], color='black', s=10)
        plt.scatter(self.mesh.x[:, :, 0], self.mesh.y[:, :, 0], color='mediumslateblue', s=10, alpha=0.5)

        segs1 = np.stack((self.mesh.nodes.x[:, :, 0], self.mesh.nodes.y[:, :, 0]), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='black', linewidths=1, alpha=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='black', linewidths=1, alpha=0.5))

        segs1 = np.stack((self.mesh.x[:, :, 0], self.mesh.y[:, :, 0]), axis=2)
        segs2 = segs1.transpose((1, 0, 2))

        plt.gca().add_collection(
            LineCollection(segs1, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))
        plt.gca().add_collection(
            LineCollection(segs2, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))

        # --------------------------------------------------------------------------------------------------------------
        plt.scatter(self.ghost.E.mesh.nodes.x, self.ghost.E.mesh.nodes.y, color='red', marker='o', s=10)
        segs1 = np.stack((self.ghost.E.mesh.nodes.x, self.ghost.E.mesh.nodes.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, alpha=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, alpha=0.5))

        plt.scatter(self.ghost.E.mesh.x, self.ghost.E.mesh.y, color='red', marker='o', s=10, alpha=0.2)
        segs1 = np.stack((self.ghost.E.mesh.x, self.ghost.E.mesh.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, linestyles='--', alpha=0.2))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, linestyles='--', alpha=0.2))

        # --------------------------------------------------------------------------------------------------------------
        plt.scatter(self.ghost.W.mesh.nodes.x, self.ghost.W.mesh.nodes.y, color='red', marker='o', s=10)
        segs1 = np.stack((self.ghost.W.mesh.nodes.x, self.ghost.W.mesh.nodes.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, alpha=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, alpha=0.5))

        plt.scatter(self.ghost.W.mesh.x, self.ghost.W.mesh.y, color='red', marker='o', s=10, alpha=0.2)
        segs1 = np.stack((self.ghost.W.mesh.x, self.ghost.W.mesh.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, linestyles='--', alpha=0.2))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, linestyles='--', alpha=0.2))

        # --------------------------------------------------------------------------------------------------------------
        plt.scatter(self.ghost.N.mesh.nodes.x, self.ghost.N.mesh.nodes.y, color='red', marker='o', s=10)
        segs1 = np.stack((self.ghost.N.mesh.nodes.x, self.ghost.N.mesh.nodes.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, alpha=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, alpha=0.5))

        plt.scatter(self.ghost.N.mesh.x, self.ghost.N.mesh.y, color='red', marker='o', s=10, alpha=0.2)
        segs1 = np.stack((self.ghost.N.mesh.x, self.ghost.N.mesh.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, linestyles='--', alpha=0.2))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, linestyles='--', alpha=0.2))

        # --------------------------------------------------------------------------------------------------------------
        plt.scatter(self.ghost.S.mesh.nodes.x, self.ghost.S.mesh.nodes.y, color='red', marker='o', s=10)
        segs1 = np.stack((self.ghost.S.mesh.nodes.x, self.ghost.S.mesh.nodes.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, alpha=0.5))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, alpha=0.5))

        plt.scatter(self.ghost.S.mesh.x, self.ghost.S.mesh.y, color='red', marker='o', s=10, alpha=0.2)
        segs1 = np.stack((self.ghost.S.mesh.x, self.ghost.S.mesh.y), axis=2)
        segs2 = segs1.transpose((1, 0, 2))
        plt.gca().add_collection(LineCollection(segs1, colors='red', linewidths=1, linestyles='--', alpha=0.2))
        plt.gca().add_collection(LineCollection(segs2, colors='red', linewidths=1, linestyles='--', alpha=0.2))

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    @property
    def Flux_EW(self):
        return self.fvm.Flux_EW

    @property
    def Flux_NS(self):
        return self.fvm.Flux_NS

    def __getitem__(self, index):
        y, x, var = index

        if self._index_in_west_ghost_block(x, y):
            return self.ghost.W.state[y, 0, var]
        elif self._index_in_east_ghost_block(x, y):
            return self.ghost.E.state[y, 0, var]
        elif self._index_in_north_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        elif self._index_in_south_ghost_block(x, y):
            return self.ghost.N.state[0, x, var]
        else:
            raise ValueError('Incorrect indexing')

    @staticmethod
    def _get_side_length(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    @staticmethod
    def _get_side_angle(pt1, pt2):
        if pt1[1] == pt2[1]:
            return np.pi / 2
        elif pt1[0] == pt2[0]:
            return 0
        else:
            return np.arctan((pt1[0] - pt2[0]) / (pt2[1] - pt1[1]))

    def _index_in_west_ghost_block(self, x, y):
        return x < 0 and 0 <= y <= self.mesh.ny

    def _index_in_east_ghost_block(self, x, y):
        return x > self.mesh.nx and 0 <= y <= self.mesh.ny

    def _index_in_south_ghost_block(self, x, y):
        return y < 0 and 0 <= x <= self.mesh.nx

    def _index_in_north_ghost_block(self, x, y):
        return y > self.mesh.ny and 0 <= x <= self.mesh.nx

    # ------------------------------------------------------------------------------------------------------------------
    # Grid methods

    # Build connectivity with neighbor blocks
    def connect(self, NeighborE: QuadBlock,
                      NeighborW: QuadBlock,
                      NeighborN: QuadBlock,
                      NeighborS: QuadBlock) -> None:

        self.neighbors = Neighbors(E=NeighborE, W=NeighborW, N=NeighborN, S=NeighborS)

    def get_east_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the WEST boundary condition for this block's EAST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second last column on the block's state will be returned.
            - if nghost = 2, the second and third last column on the block's state will be returned.
            - general case, return -(nghost + 1):-1 columns

        This is illustrated in the figure below:

            nghost = 1: return second last column (-2)

                       ...        -3         -2         -1
                                       .............
            O----------O----------O----.-----O-----.----O
            |          |          |    .     |     .    |\
            |          |          |    .     |     .    |-O
            |          |          |    .     |     .    | |\
            O----------O----------O----.-----O-----.----O |-O
            |          |          |    .     |     .    |\| |\
            |          |          |    .     |     .    |-O |-O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
            |          |          |    .     |     .    |\| |\|
            |          |          |    .     |     .    |-O |-O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
            |          |          |    .     |     .    |\| |\|
            |          |          |    .     |     .    |-O | O
            |          |          |    .     |     .    | |\| |
            O----------O----------O----.-----O-----.----O |-O |
             \|         \|         \|   .     \|    .    \| |\|
              O----------O----------O----.-----O-----.----O |-O
               \|         \|         \|   .     \|    .    \| |
                O----------O----------O----.-----O-----.----O |
                 \|         \|         \|   .     \|    .    \|
                  O----------O----------O----.-----O-----.----O
                                              .............

            nghost = 2: return third and second last colums (-3, -2)

                       ...        -3         -2         -1
                            ........................
            O----------O----.-----O----------O-----.----O
            |          |    .     |          |     .    |\
            |          |    .     |          |     .    |-O
            |          |    .     |          |     .    | |\
            O----------O----.-----O----------O-----.----O |-O
            |          |    .     |          |     .    |\| |\
            |          |    .     |          |     .    |-O |-O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
            |          |    .     |          |     .    |\| |\|
            |          |    .     |          |     .    |-O |-O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
            |          |    .     |          |     .    |\| |\|
            |          |    .     |          |     .    |-O | O
            |          |    .     |          |     .    | |\| |
            O----------O----.-----O----------O-----.----O |-O |
             \|         \|   .     \|         \|    .    \| |\|
              O----------O----.-----O----------O-----.----O |-O
               \|         \|   .     \|         \|    .    \| |
                O----------O----.-----O----------O-----.----O |
                 \|         \|   .     \|         \|    .    \|
                  O----------O----.-----O----------O-----.----O
                                   ........................
        """

        return self.state.U[:, -self.mesh.nghost:, :].copy()

    def get_west_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the EAST boundary condition for this block's WEST neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second column on the block's state will be returned.
            - if nghost = 2, the second and third column on the block's state will be returned.
            - general case, return 1:(nghost + 1) columns

        This is illustrated in the figure below:

            nghost = 1: return second column (1)

            0          1          2          3          4
                 ............
            O----.-----O----.-----O----------O----------O
            |    .     |    .     |          |          |\
            |    .     |    .     |          |          |-O
            |    .     |    .     |          |          | |\
            O----.-----O----.-----O----------O----------O |-O
            |    .     |    .     |          |          |\| |\
            |    .     |    .     |          |          |-O |-O
            |    .     |    .     |          |          | |\| |
            O----.-----O----.-----O----------O----------O |-O |
            |    .     |    .     |          |          |\| |\|
            |    .     |    .     |          |          |-O |-O
            |    .     |    .     |          |          | |\| |
            O----.-----O----.-----O----------O----------O |-O |
            |    .     |    .     |          |          |\| |\|
            |    .     |    .     |          |          |-O | O
            |    .     |    .     |          |          | |\| |
            O----.-----O----------O----------O----------O |-O |
             \|   .     \|   .     \|         \|         \| |\|
              O----.-----O----------O----------O----------O |-O
               \|   .     \|   .     \|         \|         \| |
                O----.-----O----------O----------O----------O |
                 \|   .     \|   .     \|         \|         \|
                  O----.-----O----.-----O----------O----------O
                        ............

            nghost = 2: return second and third colums (1, 2)

            0          1          2          3          4
                 .......................
            O----.-----O----------O----.-----O----------O
            |    .     |          |    .     |          |\
            |    .     |          |    .     |          |-O
            |    .     |          |    .     |          | |\
            O----.-----O----------O----.-----O----------O |-O
            |    .     |          |    .     |          |\| |\
            |    .     |          |    .     |          |-O |-O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
            |    .     |          |    .     |          |\| |\|
            |    .     |          |    .     |          |-O |-O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
            |    .     |          |    .     |          |\| |\|
            |    .     |          |    .     |          |-O | O
            |    .     |          |    .     |          | |\| |
            O----.-----O----------O----.-----O----------O |-O |
             \|   .     \|         \|   .     \|         \| |\|
              O----.-----O----------O----.-----O----------O |-O
               \|   .     \|         \|   .     \|         \| |
                O----.-----O----------O----.-----O----------O |
                 \|   .     \|         \|   .     \|         \|
                  O----.-----O----------O----.-----O----------O
                        .......................
        """

        return self.state.U[:, :self.mesh.nghost, :].copy()

    def get_north_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the SOUTH boundary condition for this block's NORTH neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second last row on the block's state will be returned.
            - if nghost = 2, the second and third last rows on the block's state will be returned.
            - general case, return -(nghost + 1):-1 rows

        This is illustrated in the figure below:

            nghost = 1: return second last row (-2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
          ............................................... O
          . |          |          |          |          |.|\
          . O----------O----------O----------O----------O . O
          . |          |          |          |          | |.|\
          ............................................... O . O
            |          |          |          |          |.|\|.|
            O----------O----------O----------O----------O . O .
            |          |          |          |          |\|.|\|.
            |          |          |          |          |-O . O.
            |          |          |          |          | |\|.|.
            O----------O----------O----------O----------O |-O ..
            |          |          |          |          |\| |\|.
            |          |          |          |          |-O | O
            |          |          |          |          | |\| |
            O----------O----------O----------O----------O |-O |
             \|         \|         \|         \|         \| |\|
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O


            nghost = 2: return second and third last rows (-3, -2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
          ...............................................-O
          . |          |          |          |          |.|\
          . O----------O----------O----------O----------O . O
          . |          |          |          |          |\|.|\
          . |          |          |          |          |-O . O
          . |          |          |          |          | |\|.|
          . O----------O----------O----------O----------O | O .
          . |          |          |          |          |\| |\|.
          ...............................................-O | O.
            |          |          |          |          |.|\|.|.
            O----------O----------O----------O----------O .-O |.
            |          |          |          |          |\|.|\|.
            |          |          |          |          |-O . O.
            |          |          |          |          | |\|.|.
            O----------O----------O----------O----------O |-O ..
             \|         \|         \|         \|         \| |\|.
              O----------O----------O----------O----------O |-O
               \|         \|         \|         \|         \| |
                O----------O----------O----------O----------O |
                 \|         \|         \|         \|         \|
                  O----------O----------O----------O----------O
        """

        return self.state.U[-self.mesh.nghost:, :, :].copy()

    def get_south_ghost(self) -> np.ndarray:
        """
        Return the solution data used to build the NORTH boundary condition for this block's SOUTH neighbor. The shape of
        the required data is dependent on the number of ghost blocks selected in the input file (nghost). For example:

            - if nghost = 1, the second row on the block's state will be returned.
            - if nghost = 2, the second and third rows on the block's state will be returned.
            - general case, return 1:(nghost + 1) rows

        This is illustrated in the figure below:

            nghost = 1: return second row (1)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
            |          |          |          |          |-O
            |          |          |          |          | |\
            O----------O----------O----------O----------O |-O
            |          |          |          |          |\| |\
            |          |          |          |          |-O |-O
            |          |          |          |          | |\| |
            O----------O----------O----------O----------O |-O |
            |          |          |          |          |\| |\|
          ...............................................-O |-O
          . |          |          |          |          |.|\| |
          . O----------O----------O----------O----------O .-O |
          . |          |          |          |          |\|.|\|
          ...............................................-O .-O
            |          |          |          |          |.|\|.|
            O----------O----------O----------O----------O .-O .
             \|         \|         \|         \|         \|.|\|.
              O----------O----------O----------O----------O .-O.
               \|         \|         \|         \|         \|.|.
                O----------O----------O----------O----------O ..
                 \|         \|         \|         \|         \|.
                  O----------O----------O----------O----------O


            nghost = 2: return second and third rows (1, 2)

            0          1          2          3          4

            O----------O----------O----------O----------O
            |          |          |          |          |\
            |          |          |          |          |-O
            |          |          |          |          | |\
            O----------O----------O----------O----------O |-O
            |          |          |          |          | | |\
          ...............................................-O |-O
          . |          |          |          |          |.|\| |
          . O----------O----------O----------O----------O .-O |
          . |          |          |          |          |\|.|\|
          . |          |          |          |          |-O .-O
          . |          |          |          |          | |\|.|
          . O----------O----------O----------O----------O |-O .
          . |          |          |          |          |\| |\|.
          ...............................................-O |-O.
            |          |          |          |          |.|\| |.
            O----------O----------O----------O----------O .-O |.
             \|         \|         \|         \|         \|.|\|.
              O----------O----------O----------O----------O .-O.
               \|         \|         \|         \|         \|.|.
                O----------O----------O----------O----------O ..
                 \|         \|         \|         \|         \|.
                  O----------O----------O----------O----------O
        """

        return self.state.U[:self.mesh.nghost, :, :].copy()

    def get_east_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the east edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: ny * 1 * 4 numpy ndarray with the east edge data

        """

        return self.state.U[:, -1:, :].copy()

    def get_west_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the west edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: ny * 1 * 4 numpy ndarray with the west edge data

        """

        return self.state.U[:, 0:1, :].copy()

    def get_north_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the north edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: 1 * nx * 4 numpy ndarray with the north edge data

        """

        return self.state.U[-1:, :, :].copy()

    def get_south_edge(self) -> np.ndarray:
        """
        Returns data from the Block's state along the south edge of the mesh.

        Parameters:
            N.A

        Return:
            - np.ndarray: 1 * nx * 4 numpy ndarray with the south edge data

        """

        return self.state.U[0:1, :, :].copy()

    def row(self, index: int) -> np.ndarray:
        return self.state.U[index:(index + 1), :, :]

    def col(self, index: int) -> np.ndarray:
        return self.state.U[None, :, index, :]

    def fullrow(self, index: int) -> np.ndarray:
        return np.concatenate((self.ghost.W[index, None, :],
                               self.row(index),
                               self.ghost.E[index, None, :]),
                               axis=1)

    def fullcol(self, index: int) -> np.ndarray:
        return np.concatenate((self.ghost.S[None, 0, index, None, :],
                               self.col(index),
                               self.ghost.N[None, 0, index, None, :]),
                               axis=1)

    def row_copy(self, index: int) -> np.ndarray:
        return self.row(index).copy()

    def col_copy(self, index: int) -> np.ndarray:
        return self.col(index).copy()

    def fullrow_copy(self, index: int) -> np.ndarray:
        return self.fullrow(index).copy()

    def fullcol_copy(self, index: int) -> np.ndarray:
        return self.fullrow(index).copy()

    def get_interface_values(self, reconstruction_type: str = None):

        if self.inputs.interface_interpolation == 'arithmetic_average':
            interfaceEW, interfaceNS = self.get_interface_values_arithmetic(reconstruction_type)
            return interfaceEW, interfaceNS
        else:
            raise ValueError('Interface Interpolation method is not defined.')

    def get_interface_values_arithmetic(self, reconstruction_type: str = None) -> [np.ndarray]:

        # Concatenate mesh state and ghost block states
        if reconstruction_type == 'Primitive':
            _W = self.state.to_primitive_vector()

            catx = np.concatenate((self.ghost.W.state.to_primitive_vector(),
                                   _W,
                                   self.ghost.E.state.to_primitive_vector()),
                                  axis=1)

            caty = np.concatenate((self.ghost.N.state.to_primitive_vector(),
                                   _W,
                                   self.ghost.S.state.to_primitive_vector()),
                                  axis=0)

        elif reconstruction_type == 'Conservative' or not reconstruction_type:

            catx = np.concatenate((self.ghost.W.state.U,
                                   self.state.U,
                                   self.ghost.E.state.U),
                                  axis=1)

            caty = np.concatenate((self.ghost.S.state.U,
                                   self.state.U,
                                   self.ghost.N.state.U),
                                  axis=0)

        else:
            raise ValueError('Undefined reconstruction type')

        # Compute arithmetic mean
        interfaceEW = 0.5 * (catx[:, 1:, :] + catx[:, :-1, :])
        interfaceNS = 0.5 * (caty[1:, :, :] + caty[:-1, :, :])

        return interfaceEW, interfaceNS

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    def update(self, dt: float) -> None:
        """
        Updates the solution state stored in the current block. Also includes any pre-processing needed prior to the
        calculation of the state updates, such as preconditioning, etc. The core operation in this method is calling the
        _time_integrator class, which steps the solution state through time by an amount of dt seconds.

        Parameters:
            - dt (float): Time step

        Returns:
            - N.A
        """

        self._time_integrator(self, dt)

    def get_flux(self) -> None:
        self.fvm.get_flux(self)

    def dUdt(self) -> None:
        return self.fvm.dUdt(self)

    def set_BC(self) -> None:
        self.ghost.E.set_BC()
        self.ghost.W.set_BC()
        self.ghost.N.set_BC()
        self.ghost.S.set_BC()

    # ------------------------------------------------------------------------------------------------------------------
    # Gradient methods

    def drho_dx(self) -> np.ndarray:
        """
        Calculate the derivative of density with respect to the x direction. This is equivalent of returning
        gradx[:, :, 0], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - drho_dx(np.ndarray): Derivative of rho with respect to the x direction.
        """

        return self.gradx[:, :, 0, None]

    def du_dx(self) -> np.ndarray:
        """
        Calculate the derivative of u with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.gradx`. To compute this gradient, utilize the chain rule on drhou_dx:

        \\frac{\\partial(\\rho u)}{\\partial x} = \\rho \\frac{\\partial u}{\\partial x} +
        u \\frac{\\partial \\rho}{\\partial x},

        and rearrange to compute du_dx:

        \\frac{\\partial u}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho u)}{\\partial x} -
        u \\frac{\\partial \\rho}{\\partial x} \\right).

        Parameters:
            - N.A

        Returns:
            - du_dx (np.ndarray): Derivative of u with respect to the x direction.
        """

        return (self.gradx[:, :, 1, None] - self.state.u() * self.drho_dx()) / self.state.rho

    def dv_dx(self) -> np.ndarray:
        """
        Calculate the derivative of v with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.gradx`. To compute this gradient, utilize the chain rule on drhov_dx:

        \\frac{\\partial(\\rho v)}{\\partial x} = \\rho \\frac{\\partial v}{\\partial x} +
        v \\frac{\\partial \\rho}{\\partial x},

        and rearrange to compute dv_dx:

        \\frac{\\partial v}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho v)}{\\partial x} -
        v \\frac{\\partial \\rho}{\\partial x} \\right).

        Parameters:
            - N.A

        Returns:
            - dv_dx (np.ndarray): Derivative of v with respect to the x direction.
        """

        return (self.gradx[:, :, 2, None] - self.state.v() * self.drho_dx()) / self.state.rho

    def de_dx(self) -> np.ndarray:
        """
        Calculate the derivative of energy with respect to the x direction. This is equivalent of returning
        gradx[:, :, 3], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - de_dx (np.ndarray): Derivative of e with respect to the x direction.
        """

        return self.gradx[:, :, -1, None]

    def drho_dy(self) -> np.ndarray:
        """
        Calculate the derivative of density with respect to the y direction. This is equivalent of returning
        grady[:, :, 0], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - drho_dy (np.ndarray): Derivative of rho with respect to the y direction.
        """
        return self.grady[:, :, 0, None]

    def du_dy(self) -> np.ndarray:
        """
        Calculate the derivative of u with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.grady`. To compute this gradient, utilize the chain rule on drhou_dy:

        \\frac{\\partial(\\rho u)}{\\partial y} = \\rho \\frac{\\partial u}{\\partial y} +
        u \\frac{\\partial \\rho}{\\partial y},

        and rearrange to compute du_dy:

        \\frac{\\partial u}{\\partial y} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho u)}{\\partial y} -
        u \\frac{\\partial \\rho}{\\partial y} \\right).

        Parameters:
            - N.A

        Returns:
            - du_dy (np.ndarray): Derivative of u with respect to the y direction.
        """

        return (self.grady[:, :, 1, None] - self.state.u() * self.drho_dx()) / self.state.rho

    def dv_dy(self) -> np.ndarray:
        """
        Calculate the derivative of v with respect to the x direction. This is done by applying the chain rule to the
        available x-direction gradients. The derivatives of the conservative variables (rho, rho*u, rho*v, e) are
        available under `self.refBLK.grady`. To compute this gradient, utilize the chain rule on drhov_dy:

        \\frac{\\partial(\\rho v)}{\\partial y} = \\rho \\frac{\\partial v}{\\partial y} +
        v \\frac{\\partial \\rho}{\\partial y},

        and rearrange to compute dv_dx:

        \\frac{\\partial v}{\\partial x} = \\frac{1}{\\rho}\\left(\\frac{\\partial(\\rho v)}{\\partial y} -
        v \\frac{\\partial \\rho}{\\partial y} \\right).

        Parameters:
            - N.A

        Returns:
            - dv_dy (np.ndarray): Derivative of v with respect to the y direction.
        """

        return (self.grady[:, :, 2, None] - self.state.v() * self.drho_dx()) / self.state.rho

    def de_dy(self) -> np.ndarray:
        """
        Calculate the derivative of energy with respect to the y direction. This is equivalent of returning
        grady[:, :, 3], and no further calculations are needed.

        Parameters:
            - N.A

        Returns:
            - de_dy (np.ndarray): Derivative of e with respect to the y direction.
        """

        return self.grady[:, :, -1, None]

    # Test test
