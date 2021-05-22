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
from typing import Union
from pyHype.states.states import ConservativeState
from pyHype.mesh.mesh_inputs import BlockDescription
from pyHype.input.input_file_builder import ProblemInput
import pyHype.solvers.time_integration.explicit_runge_kutta as erk

from pyHype.fvm import SecondOrderGreenGauss

from pyHype.blocks.boundary import BoundaryBlockEast, \
                          BoundaryBlockWest, \
                          BoundaryBlockSouth,\
                          BoundaryBlockNorth,\
                          BoundaryBlock


class Vertices:
    def __init__(self, NE: tuple[Union[float, int], Union[float, int]],
                       NW: tuple[Union[float, int], Union[float, int]],
                       SE: tuple[Union[float, int], Union[float, int]],
                       SW: tuple[Union[float, int], Union[float, int]]) -> None:
        self.NW = NW
        self.NE = NE
        self.SW = SW
        self.SE = SE


class Neighbors:
    def __init__(self, E: 'QuadBlock' = None,
                       W: 'QuadBlock' = None,
                       N: 'QuadBlock' = None,
                       S: 'QuadBlock' = None) -> None:
        self.E = E
        self.W = W
        self.N = N
        self.S = S


class BoundaryBlockContainer:
    def __init__(self, E: 'BoundaryBlock' = None,
                       W: 'BoundaryBlock' = None,
                       N: 'BoundaryBlock' = None,
                       S: 'BoundaryBlock' = None) -> None:
        self.E = E
        self.W = W
        self.N = N
        self.S = S


class NormalVector:
    def __init__(self, theta):
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


class Mesh:
    def __init__(self, inputs, mesh_data):
        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.vertices = Vertices(NW=mesh_data.NW,
                                 NE=mesh_data.NE,
                                 SW=mesh_data.SW,
                                 SE=mesh_data.SE)

        X, Y = np.meshgrid(np.linspace(self.vertices.NW[0], self.vertices.NE[0], self.nx),
                           np.linspace(self.vertices.SE[1], self.vertices.NE[1], self.ny))
        self.x = X
        self.y = Y

        self.Lx    = self.vertices.NE[0] - self.vertices.NW[0]
        self.Ly    = self.vertices.NE[1] - self.vertices.SE[1]

        self.dx     = self.Lx / (self.nx + 1)
        self.dy     = self.Lx / (self.nx + 1)


class Blocks:
    def __init__(self, inputs):
        self.inputs = inputs
        self._number_of_blocks = None
        self._blocks = {}
        self._connectivity = {}

        self.build()

    @property
    def blocks(self):
        return self._blocks

    def add(self, block) -> None:
        self._blocks[block.global_nBLK] = block

    def get(self, block_idx: int):
        return self._blocks[block_idx]

    def update(self, dt) -> None:
        for block in self._blocks.values():
            block.update(dt)

    def set_BC(self) -> None:
        for block in self._blocks.values():
            block.set_BC()

    def update_BC(self) -> None:
        for block in self._blocks.values():
            block.update_BC()

    def build(self) -> None:
        mesh_inputs = self.inputs.mesh_inputs

        for BLK_data in mesh_inputs.values():
            self.add(QuadBlock(self.inputs, BLK_data))

        self._number_of_blocks = len(self._blocks)

        for global_nBLK, block in self._blocks.items():
            Neighbor_E_idx = mesh_inputs.get(block.global_nBLK).NeighborE
            Neighbor_W_idx = mesh_inputs.get(block.global_nBLK).NeighborW
            Neighbor_N_idx = mesh_inputs.get(block.global_nBLK).NeighborN
            Neighbor_S_idx = mesh_inputs.get(block.global_nBLK).NeighborS

            block.connect(NeighborE=self._blocks[Neighbor_E_idx] if Neighbor_E_idx != 0 else None,
                          NeighborW=self._blocks[Neighbor_W_idx] if Neighbor_W_idx != 0 else None,
                          NeighborN=self._blocks[Neighbor_N_idx] if Neighbor_N_idx != 0 else None,
                          NeighborS=self._blocks[Neighbor_S_idx] if Neighbor_S_idx != 0 else None)

    def print_connectivity(self) -> None:
        for _, block in self._blocks.items():
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
        self._mesh              = Mesh(inputs, block_data)
        self._state             = ConservativeState(inputs, nx=inputs.nx, ny=inputs.ny)
        self.global_nBLK        = block_data.nBLK
        self.boundaryBLK        = None
        self.neighbors          = None

        vert = self._mesh.vertices

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

        if fvm == 'FirstOrderUnlimited':
            self._finite_volume_method = FirstOrderUnlimited(self.inputs, self.global_nBLK)
        elif fvm == 'FirstOrderLimited':
            self._finite_volume_method = FirstOrderUnlimited(self.inputs, self.global_nBLK)
        elif fvm == 'SecondOrderGreenGauss':
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
        return x < 0 and 0 <= y <= self._mesh.ny

    def _index_in_east_boundary_block(self, x, y):
        return x > self._mesh.nx and 0 <= y <= self._mesh.ny

    def _index_in_south_boundary_block(self, x, y):
        return y < 0 and 0 <= x <= self._mesh.nx

    def _index_in_north_boundary_block(self, x, y):
        return y > self._mesh.ny and 0 <= x <= self._mesh.nx

    @property
    def vertices(self):
        return self.vertices

    @property
    def state(self):
        return self._state

    @property
    def mesh(self):
        return self._mesh

    # ------------------------------------------------------------------------------------------------------------------
    # Grid methods

    # Build connectivity with neigbor blocks
    def connect(self, NeighborE: 'QuadBlock',
                      NeighborW: 'QuadBlock',
                      NeighborN: 'QuadBlock',
                      NeighborS: 'QuadBlock') -> None:

        self.neighbors = Neighbors(E=NeighborE, W=NeighborW, N=NeighborN, S=NeighborS)

    def get_east_edge(self) -> np.ndarray:
        return self.state.U[None, -1, :]

    def get_west_edge(self) -> np.ndarray:
        return self.state.U[None, 0, :]

    def get_north_edge(self) -> np.ndarray:
        return self.state.U[-1, None, :]

    def get_south_edge(self) -> np.ndarray:
        return self.state.U[0, None, :]

    def row(self, index: int) -> np.ndarray:
        return self._state.U[index, None, :]

    def fullrow(self, index: int) -> np.ndarray:
        return np.concatenate((self.boundaryBLK.W[index, None, :],
                               self.row(index),
                               self.boundaryBLK.E[index, None, :]),
                               axis=1)

    def col(self, index: int) -> np.ndarray:
        return self._state.U[None, :, index, :]

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
        self.update_BC()

    def update_BC(self) -> None:
        self.boundaryBLK.E.set()
        self.boundaryBLK.W.set()
        self.boundaryBLK.N.set()
        self.boundaryBLK.S.set()
