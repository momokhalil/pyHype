import numpy as np
from typing import Union
from abc import abstractmethod, ABC
from pyHype.states.states import ConservativeState
from pyHype.mesh.mesh_builder import BlockDescription
from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm import FirstOrderUnlimited, SecondOrderLimited


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

        self.Lx    = self.vertices.NE[0] - self.vertices.NW[0]
        self.Ly    = self.vertices.NE[1] - self.vertices.SE[1]
        self.nx     = inputs.nx
        self.ny     = inputs.ny
        self.dx     = self.Lx / (self.nx + 1)
        self.dy     = self.Lx / (self.nx + 1)

        X, Y        = np.meshgrid(np.linspace(self.vertices.NW[0], self.vertices.NE[0], self.nx),
                                  np.linspace(self.vertices.SE[1], self.vertices.NE[1], self.ny))
        self._x     = X
        self._y     = Y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


# QuadBlock Class Definition
class QuadBlock:
    def __init__(self, inputs: ProblemInput, block_data: BlockDescription) -> None:

        self.inputs             = inputs
        self._mesh              = Mesh(inputs, block_data)
        self._state             = ConservativeState(inputs, inputs.n)
        self.global_nBLK        = block_data.nBLK
        self.boundary_blocks    = None
        self.neighbors          = None

        # Set finite volume method
        fvm = self.inputs.finite_volume_method

        if fvm == 'FirstOrderUnlimited':
            self._finite_volume_method = FirstOrderUnlimited(self.inputs, self.global_nBLK)
        elif fvm == 'FirstOrderLimited':
            self._finite_volume_method = FirstOrderUnlimited(self.inputs, self.global_nBLK)
        elif fvm == 'SecondOrderLimited':
            self._finite_volume_method = SecondOrderLimited(self.inputs, self.global_nBLK)
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

        # Construct indices to access column-wise elements on the mesh
        self.col_idx = np.ones((4 * self._mesh.ny), dtype=np.int32)

        for i in range(1, self._mesh.ny + 1):
            self.col_idx[4 * i - 4:4 * i] = np.arange(4 * self._mesh.nx * (i - 1) - 4, 4 * self._mesh.nx * (i - 1))

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
        return self.boundary_blocks.E.from_ref_U()

    def get_west_edge(self) -> np.ndarray:
        return self.boundary_blocks.W.from_ref_U()

    def get_north_edge(self) -> np.ndarray:
        return self.boundary_blocks.N.from_ref_U()

    def get_south_edge(self) -> np.ndarray:
        return self.boundary_blocks.S.from_ref_U()

    def row(self, index: int) -> np.ndarray:
        return self._state.U[4*self._mesh.nx*(index - 1):4*self._mesh.nx*index]

    def col(self, index: int) -> np.ndarray:
        return self._state.U[self.col_idx + 4*index]

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    # Update solution state
    def update(self, dt) -> None:
        self._time_integrator(dt)

    # Explicit Euler time stepping
    def explicit_euler(self, dt) -> None:

        # First stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # Update block state vector
        self._state.U += dt * Rx / self._mesh.dx + dt * Ry / self._mesh.dy
        # Update block state variables
        self._state.set_vars_from_state()
        # Update state BC
        self.update_BC()

    # RK2 time stepping
    def RK2(self, dt) -> None:

        # Save state for final stage
        u = self._state.U

        # First stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # Update block state vector
        self._state.U += 0.5 * (dt * Rx / self._mesh.dx + dt * Ry / self._mesh.dy)
        # Update block state variables
        self._state.set_vars_from_state()
        # Update state BC
        self.update_BC()

        # Second stage ##############################################################

        # Get residuals
        Rx, Ry = self.get_residual()
        # Update block state vector
        self._state.U = u + dt * Rx / self._mesh.dx + dt * Ry / self._mesh.dy
        # Update block state variables
        self._state.set_vars_from_state()
        # Update state BC
        self.update_BC()

    # RK3 TVD time stepping
    def RK3TVD(self, dt) -> None:
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
        self.inputs = inputs
        self._idx_from_U = None
        self._state = None
        self._type = type_
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.ref_BLK = ref_BLK

    def __getitem__(self, index):
        return self.state.U[4 * index - 4:4 * index]

    @property
    def state(self):
        return self._state

    @abstractmethod
    def set(self):
        pass

    def from_ref_U(self):
        return self.ref_BLK.state.U[self._idx_from_U]

class BoundaryBlockNorth(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._idx_from_U = slice(4 * self.nx * (self.ny - 1), 4 * self.nx * self.ny)
        self._state = ConservativeState(inputs, self.nx)

    def set(self) -> None:
        if self._type   == 'Outflow':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = self.ref_BLK.neighbors.N.get_south_edge()
        elif self._type == 'Reflection':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockSouth(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._idx_from_U = slice(0, 4 * self.nx)
        self._state = ConservativeState(inputs, self.nx)

    def set(self) -> None:
        if self._type   == 'Outflow':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = self.ref_BLK.neighbors.S.get_north_edge()
        elif self._type == 'Reflection':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockEast(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(inputs, self.ny)

        for j in range(1, self.ny + 1):
            iF = 4 * self.nx * j - 4
            iE = 4 * self.nx * j
            self._idx_from_U[4 * j - 4:4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self) -> None:
        if self._type   == 'Outflow':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = self.ref_BLK.neighbors.E.get_west_edge()
        elif self._type == 'Reflection':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1

class BoundaryBlockWest(BoundaryBlock):
    def __init__(self, inputs, type_, ref_BLK):
        super().__init__(inputs, type_, ref_BLK)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(inputs, self.ny)

        for j in range(1, self.ny + 1):
            iF = (4 * j - 4) + 4 * (j - 1) * (self.nx - 1)
            iE = (4 * j - 0) + 4 * (j - 1) * (self.ny - 1)
            self._idx_from_U[4 * j - 4: 4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self) -> None:
        if self._type == 'Outflow':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = self.ref_BLK.neighbors.W.get_east_edge()
        elif self._type == 'Reflection':
            self._state.U = self.ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1
