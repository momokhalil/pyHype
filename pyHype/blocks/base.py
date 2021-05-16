import time
import numpy as np
from typing import Union
from abc import abstractmethod, ABC
from pyHype.states.states import ConservativeState
from pyHype.mesh.mesh_inputs import BlockDescription
from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm import FirstOrderUnlimited, SecondOrderGreenGauss


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


class BoundaryBlocks:
    def __init__(self, E: 'BoundaryBlock' = None,
                       W: 'BoundaryBlock' = None,
                       N: 'BoundaryBlock' = None,
                       S: 'BoundaryBlock' = None) -> None:
        self.E = E
        self.W = W
        self.N = N
        self.S = S


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


class Mesh:
    def __init__(self, inputs, mesh_data):
        self.inputs = inputs
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
        self.nx     = inputs.nx
        self.ny     = inputs.ny
        self.dx     = self.Lx / (self.nx + 1)
        self.dy     = self.Lx / (self.nx + 1)




# QuadBlock Class Definition
class QuadBlock:
    def __init__(self, inputs: ProblemInput, block_data: BlockDescription) -> None:

        self.inputs             = inputs
        self._mesh              = Mesh(inputs, block_data)
        self._state             = ConservativeState(inputs, nx=inputs.nx, ny=inputs.ny)
        self.global_nBLK        = block_data.nBLK
        self.boundary_blocks    = None
        self.neighbors          = None

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

        if time_integrator      == 'ExplicitEuler':
            self._time_integrator = self.explicit_euler
        elif time_integrator    == 'RK2':
            self._time_integrator = self.RK2
        elif time_integrator    == 'RK3TVD':
            self._time_integrator = self.RK3TVD
        elif time_integrator    == 'RK4':
            self._time_integrator = self.RK4
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # Build boundary blocks
        self.boundary_blocks = BoundaryBlocks(E=BoundaryBlockEast(self.inputs, type_=block_data.BCTypeE, ref_BLK=self),
                                              W=BoundaryBlockWest(self.inputs, type_=block_data.BCTypeW, ref_BLK=self),
                                              N=BoundaryBlockNorth(self.inputs, type_=block_data.BCTypeN, ref_BLK=self),
                                              S=BoundaryBlockSouth(self.inputs, type_=block_data.BCTypeS, ref_BLK=self))

    def __getitem__(self, index):
        y, x, var = index

        if self._index_in_west_boundary_block(x, y):
            return self.boundary_blocks.W.state[y, 0, var]
        elif self._index_in_east_boundary_block(x, y):
            return self.boundary_blocks.E.state[y, 0, var]
        elif self._index_in_north_boundary_block(x, y):
            return self.boundary_blocks.N.state[0, x, var]
        elif self._index_in_south_boundary_block(x, y):
            return self.boundary_blocks.N.state[0, x, var]
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
        return self._mesh.vertices

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
        return np.concatenate((self.boundary_blocks.W[index, None, :],
                               self.row(index),
                               self.boundary_blocks.E[index, None, :]),
                               axis=1)

    def col(self, index: int) -> np.ndarray:
        return self._state.U[None, :, index, :]

    def fullcol(self, index: int) -> np.ndarray:
        return np.concatenate((self.boundary_blocks.S[None, 0, index, None, :],
                               self.col(index),
                               self.boundary_blocks.N[None, 0, index, None, :]),
                               axis=1)

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    # Update solution state
    def update(self, dt) -> None:
        self._time_integrator(dt)

    # Explicit Euler time stepping
    def explicit_euler(self, dt) -> None:

        # Save inial state
        U_initial = self._state.U

        # First stage ##############################################################
        print(dt)
        # Get residuals
        Rx, Ry = self.get_residual()
        # First update vector
        K1 = U_initial + dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)

        # Update block state vector
        self._state.update(K1)
        # Update state BC
        self.update_BC()

    # RK2 time stepping
    def RK2(self, dt) -> None:

        # Save inial state
        U_initial = self._state.U

        # First stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # First update vector
        K1 = U_initial + 0.5 * dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)
        # Update block state vector
        self._state.update(K1)
        # Update state BC
        self.update_BC()

        # Second stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # First update vector
        K2 = U_initial + dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)
        # Update block state vector
        self._state.update(K2)
        # Update state BC
        self.update_BC()

    # RK3 TVD time stepping
    def RK3TVD(self, dt) -> None:

        # Save inial state
        U_initial = self._state.U

        # First stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # First update vector
        K1 = U_initial + dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)
        # Update block state vector
        self._state.update(K1)
        # Update state BC
        self.update_BC()

        # Second stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # Second update vector
        K2 = 0.75 * U_initial +     \
             0.25 * K1 +            \
             0.25 * dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)
        # Update block state vector
        self._state.update(K2)
        # Update state BC
        self.update_BC()

        # Third stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # Third update vector
        K3 = (1/3) * U_initial + \
             (2/3) * K2 +        \
             (2/3) * dt * (Rx / self._mesh.dx + Ry / self._mesh.dy)
        # Update block state vector
        self._state.update(K3)
        # Update state BC
        self.update_BC()

        pass

    # RK4 time stepping
    def RK4(self, dt) -> None:
        pass

    # Calculate residuals in x and y directions
    def get_residual(self):
        self._finite_volume_method.get_flux(self)
        return -self._finite_volume_method.Flux_X, -self._finite_volume_method.Flux_Y

    def set_BC(self) -> None:
        self.update_BC()

    def update_BC(self) -> None:
        self.boundary_blocks.E.set()
        self.boundary_blocks.W.set()
        self.boundary_blocks.N.set()
        self.boundary_blocks.S.set()


class BoundaryBlock(ABC):
    def __init__(self, inputs, type_: str, ref_BLK: 'QuadBlock'):

        self._type = type_
        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.ref_BLK = ref_BLK

        self._state = None

    def __getitem__(self, index):
        #x, y, var = index
        return self.state.U[index]

    @property
    def state(self):
        return self._state

    def set(self) -> None:
        if self._type == 'None':
            self.set_BC_none()
        elif self._type == 'Outflow':
            self.set_BC_outflow()
        elif self._type == 'Reflection':
            self.set_BC_reflection()

    @abstractmethod
    def from_ref_U(self):
        pass

    @abstractmethod
    def set_BC_none(self):
        pass

    @abstractmethod
    def set_BC_outflow(self):
        pass

    @abstractmethod
    def set_BC_reflection(self):
        pass


class BoundaryBlockNorth(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=self.nx, ny=1)

    def from_ref_U(self):
        return self.ref_BLK.state.U[-1, :, :].reshape(1, self.inputs.nx, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.N.get_south_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 2] *= -1


class BoundaryBlockSouth(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=self.nx, ny=1)

    def from_ref_U(self):
        return self.ref_BLK.state.U[0, :, :].reshape(1, self.inputs.nx, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_north_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 2] *= -1


class BoundaryBlockEast(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=1, ny=self.ny)

    def from_ref_U(self):
        return self.ref_BLK.state.U[:, -1, :].reshape(-1, 1, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_west_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 1] *= -1


class BoundaryBlockWest(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._state = ConservativeState(inputs, nx=1, ny=self.ny)

    def from_ref_U(self):
        return self.ref_BLK.state.U[:, 0, :].reshape(-1, 1, 4)

    def set_BC_none(self):
        self._state.U = self.ref_BLK.neighbors.S.get_east_edge()

    def set_BC_outflow(self):
        self._state.U = self.from_ref_U()

    def set_BC_reflection(self):
        self._state.U = self.from_ref_U()
        self._state.U[:, :, 1] *= -1
