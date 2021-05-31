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

import numpy as np
from pyHype.states.states import ConservativeState
from pyHype.mesh.base import BlockDescription, Mesh
from pyHype.input.input_file_builder import ProblemInput
from pyHype.solvers.time_integration.explicit_runge_kutta import ExplicitRungeKutta as erk

from pyHype.fvm import SecondOrderGreenGauss

from pyHype.blocks.boundary import BoundaryBlockEast, \
                          BoundaryBlockWest, \
                          BoundaryBlockSouth,\
                          BoundaryBlockNorth,\
                          BoundaryBlock


class Neighbors:
    def __init__(self,
                 E: 'QuadBlock' = None,
                 W: 'QuadBlock' = None,
                 N: 'QuadBlock' = None,
                 S: 'QuadBlock' = None
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


class BoundaryBlockContainer:
    def __init__(self,
                 E: 'BoundaryBlock' = None,
                 W: 'BoundaryBlock' = None,
                 N: 'BoundaryBlock' = None,
                 S: 'BoundaryBlock' = None
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
                 theta: float):
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
    def __init__(self, inputs):
        self.inputs = inputs
        self.number_of_blocks = None
        self.blocks = {}

        self.build()

    def __call__(self, block: int):
        return self.blocks[block]

    def add(self, block) -> None:
        self.blocks[block.global_nBLK] = block

    def get(self, block: int):
        return self.blocks[block]

    def update(self, dt) -> None:
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
    def __init__(self, inputs: ProblemInput, block_data: BlockDescription) -> None:

        self.inputs             = inputs
        self.mesh               = Mesh(inputs, block_data)
        self.state              = ConservativeState(inputs, nx=inputs.nx, ny=inputs.ny)
        self.global_nBLK        = block_data.nBLK
        self.boundaryBLK        = None
        self.neighbors          = None

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


        # Set finite volume method
        fvm = self.inputs.finite_volume_method

        if fvm == 'SecondOrderGreenGauss':
            self._finite_volume_method = SecondOrderGreenGauss(self.inputs, self.global_nBLK)
        else:
            raise ValueError('Specified finite volume method has not been specialized.')

        # Set time integrator
        time_integrator = self.inputs.time_integrator

        if time_integrator      == 'ExplicitEuler1':
            self._time_integrator = erk.ExplicitEuler1(self.inputs, self)
        elif time_integrator    == 'RK2':
            self._time_integrator = erk.RK2(self.inputs, self)
        elif time_integrator    == 'Generic2':
            self._time_integrator = erk.Generic2(self.inputs, self)
        elif time_integrator    == 'Ralston2':
            self._time_integrator = erk.Ralston2(self.inputs, self)
        elif time_integrator    == 'Generic3':
            self._time_integrator = erk.Generic3(self.inputs, self)
        elif time_integrator    == 'RK3':
            self._time_integrator = erk.RK3(self.inputs, self)
        elif time_integrator    == 'RK3SSP':
            self._time_integrator = erk.RK3SSP(self.inputs, self)
        elif time_integrator    == 'Ralston3':
            self._time_integrator = erk.Ralston3(self.inputs, self)
        elif time_integrator    == 'RK4':
            self._time_integrator = erk.RK4(self.inputs, self)
        elif time_integrator    == 'Ralston4':
            self._time_integrator = erk.Ralston4(self.inputs, self)
        elif time_integrator    == 'DormandPrince5':
            self._time_integrator = erk.DormandPrince5(self.inputs, self)
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # Build boundary blocks
        self.boundaryBLK = BoundaryBlockContainer(E=BoundaryBlockEast(self.inputs, type_=block_data.BCTypeE, ref_BLK=self),
                                                  W=BoundaryBlockWest(self.inputs, type_=block_data.BCTypeW, ref_BLK=self),
                                                  N=BoundaryBlockNorth(self.inputs, type_=block_data.BCTypeN, ref_BLK=self),
                                                  S=BoundaryBlockSouth(self.inputs, type_=block_data.BCTypeS, ref_BLK=self))

    @property
    def Flux_X(self):
        return self._finite_volume_method.Flux_X

    @property
    def Flux_Y(self):
        return self._finite_volume_method.Flux_Y

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

    def __getitem__(self, index):
        y, x, var = index

        if self._index_in_west_boundary_block(x, y):
            return self.boundaryBLK.W.state[y, 0, var]
        elif self._index_in_east_boundary_block(x, y):
            return self.boundaryBLK.E.state[y, 0, var]
        elif self._index_in_north_boundary_block(x, y):
            return self.boundaryBLK.N.state[0, x, var]
        elif self._index_in_south_boundary_block(x, y):
            return self.boundaryBLK.N.state[0, x, var]
        else:
            raise ValueError('Incorrect indexing')


    def _index_in_west_boundary_block(self, x, y):
        return x < 0 and 0 <= y <= self.mesh.ny

    def _index_in_east_boundary_block(self, x, y):
        return x > self.mesh.nx and 0 <= y <= self.mesh.ny

    def _index_in_south_boundary_block(self, x, y):
        return y < 0 and 0 <= x <= self.mesh.nx

    def _index_in_north_boundary_block(self, x, y):
        return y > self.mesh.ny and 0 <= x <= self.mesh.nx

    @property
    def vertices(self):
        return self.vertices

    # ------------------------------------------------------------------------------------------------------------------
    # Grid methods

    # Build connectivity with neighbor blocks
    def connect(self, NeighborE: 'QuadBlock',
                      NeighborW: 'QuadBlock',
                      NeighborN: 'QuadBlock',
                      NeighborS: 'QuadBlock') -> None:

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

        return self.state.U[:, -(self.mesh.nghost + 1):-1, :].copy()

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

        return self.state.U[:, 1:(self.mesh.nghost + 1), :].copy()

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

        return self.state.U[-(self.mesh.nghost + 1):-1, :, :].copy()

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

        return self.state.U[1:(self.mesh.nghost + 1), :, :].copy()

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
        return self.state.U[index, None, :]

    def fullrow(self, index: int) -> np.ndarray:

        return np.concatenate((self.boundaryBLK.W[index, None, :],
                               self.row(index),
                               self.boundaryBLK.E[index, None, :]),
                               axis=1)

    def col(self, index: int) -> np.ndarray:
        return self.state.U[None, :, index, :]

    def fullcol(self, index: int) -> np.ndarray:
        return np.concatenate((self.boundaryBLK.S[None, 0, index, None, :],
                               self.col(index),
                               self.boundaryBLK.N[None, 0, index, None, :]),
                               axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    # Update solution state
    def update(self, dt) -> None:
        self._time_integrator(dt)

    def get_flux(self):
        self._finite_volume_method.get_flux(self)

    def set_BC(self) -> None:
        self.boundaryBLK.E.set_BC()
        self.boundaryBLK.W.set_BC()
        self.boundaryBLK.N.set_BC()
        self.boundaryBLK.S.set_BC()
