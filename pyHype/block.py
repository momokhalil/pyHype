import numpy as np
from typing import Union
from abc import abstractmethod
from pyHype.states import ConservativeState
from pyHype.finite_volume_methods import FirstOrderUnlimited, SecondOrderLimited

#-----------------------------------------------------------------------------------------------------------------------
# General Classes

class Vertices:
    def __init__(self, NE: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       NW: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       SE: tuple[Union[float, int], Union[float, int]] = (0, 0),
                       SW: tuple[Union[float, int], Union[float, int]] = (0, 0)) -> None:
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

class Mesh:
    def __init__(self, input_, mesh_data_):
        self._input = input_
        self.vertices = Vertices(NW=mesh_data_.NW,
                                 NE=mesh_data_.NE,
                                 SW=mesh_data_.SW,
                                 SE=mesh_data_.SE)

        self.Lx    = self.vertices.NE[0] - self.vertices.NW[0]
        self.Ly    = self.vertices.NE[1] - self.vertices.SE[1]
        self.nx     = input_.nx
        self.ny     = input_.ny
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


#-----------------------------------------------------------------------------------------------------------------------
# Solution Blocks

# QuadBlock Class Definition
class QuadBlock:
    def __init__(self, input_, mesh_data_):
        self._input             = input_
        self._mesh              = Mesh(input_, mesh_data_)
        self._state             = ConservativeState(input_, input_.n)
        self.global_nBLK        = mesh_data_.nBLK
        self.boundary_blocks    = None
        self.neighbors          = None

        # Set finite volume method
        fvm = self._input.finite_volume_method

        if fvm == 'FirstOrderUnlimited':
            self._finite_volume_method = FirstOrderUnlimited(self._input, self.global_nBLK)
        elif fvm == 'FirstOrderLimited':
            self._finite_volume_method = FirstOrderUnlimited(self._input, self.global_nBLK)
        elif fvm == 'SecondOrderLimited':
            self._finite_volume_method = SecondOrderLimited(self._input, self.global_nBLK)
        else:
            raise ValueError('Specified time marching scheme has not been specialized.')

        # Set time integrator
        time_integrator = self._input.time_integrator

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
        BC_East     = BoundaryBlockEast(self._input, mesh_data_.BCTypeE)
        BC_West     = BoundaryBlockWest(self._input, mesh_data_.BCTypeW)
        BC_North    = BoundaryBlockNorth(self._input, mesh_data_.BCTypeN)
        BC_South    = BoundaryBlockSouth(self._input, mesh_data_.BCTypeS)
        self.boundary_blocks = BoundaryBlocks(E=BC_East, W=BC_West, N=BC_North, S=BC_South)

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
        return self.boundary_blocks.E.from_ref_U(self)

    def get_west_edge(self) -> np.ndarray:
        return self.boundary_blocks.W.from_ref_U(self)

    def get_north_edge(self) -> np.ndarray:
        return self.boundary_blocks.N.from_ref_U(self)

    def get_south_edge(self) -> np.ndarray:
        return self.boundary_blocks.S.from_ref_U(self)

    # ------------------------------------------------------------------------------------------------------------------
    # Time stepping methods

    # Update solution state
    def update(self, dt):
        self._time_integrator(dt)

    # Explicit Euler time stepping
    def explicit_euler(self):
        pass

    # RK2 time stepping
    def RK2(self, dt):
        Rx, Ry = self.get_residual()
        u = self._state.U

        k1 = dt * Rx / self._mesh.dx + dt * Ry / self._mesh.dy

        self._state.U += 0.5 * k1
        self._state.set_vars_from_state()
        self.update_BC()

        Rx, Ry = self.get_residual()
        self._state.U = u + dt * Rx / self._mesh.dx + dt * Ry / self._mesh.dy
        self._state.set_vars_from_state()
        self.update_BC()

    # RK3 TVD time stepping
    def RK3TVD(self, dt):
        pass

    # RK4 time stepping
    def RK4(self, dt):
        pass

    # Calculate residuals in x and y directions
    def get_residual(self):
        self._finite_volume_method.get_flux(self)
        return -self._finite_volume_method.Flux_X, -self._finite_volume_method.Flux_Y

    def set_BC(self):
        self.update_BC()

    def update_BC(self):
        self.boundary_blocks.E.set(ref_BLK=self)
        self.boundary_blocks.W.set(ref_BLK=self)
        self.boundary_blocks.N.set(ref_BLK=self)
        self.boundary_blocks.S.set(ref_BLK=self)
        pass

class Blocks:
    def __init__(self, input_):
        self._input = input_
        self._number_of_blocks = None
        self._blocks = {}
        self._connectivity = {}

        self.build()

    @property
    def blocks(self):
        return self._blocks

    def add(self, block):
        self._blocks[block.global_nBLK] = block

    def get(self, block_idx):
        return self._blocks[block_idx]

    def update(self, dt):
        for block in self._blocks.values():
            block.update(dt)

    def set_BC(self):
        for block in self._blocks.values():
            block.set_BC()

    def update_BC(self):
        for block in self._blocks.values():
            block.update_BC()

    def build(self):
        mesh_inputs = self._input.mesh_inputs

        for BLK_data in mesh_inputs.values():
            self.add(QuadBlock(self._input, BLK_data))

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

    def print_connectivity(self):
        for _, block in self._blocks.items():
            print('-----------------------------------------')
            print('CONNECTIVITY FOR GLOBAL BLOCK: ', block.global_nBLK, '<{}>'.format(block))
            print('North: ', block.neighbors.N)
            print('South: ', block.neighbors.S)
            print('East:  ', block.neighbors.E)
            print('West:  ', block.neighbors.W)


#-----------------------------------------------------------------------------------------------------------------------
# Boundary Blocks

class BoundaryBlock:
    def __init__(self, input_, type_: str):
        self._input = input_
        self._idx_from_U = None
        self._state = None
        self._type = type_
        self.nx = input_.nx
        self.ny = input_.ny

    @property
    def state(self):
        return self._state

    @abstractmethod
    def set(self, ref_BLK):
        pass

    def from_ref_U(self, ref_BLK):
        return ref_BLK.state.U[self._idx_from_U]

class BoundaryBlockNorth(BoundaryBlock):
    def __init__(self, input_, type_):
        super().__init__(input_, type_)
        self._idx_from_U = slice(4 * self.nx * (self.ny - 1), 4 * self.nx * self.ny)
        self._state = ConservativeState(input_, self.nx)

    def set(self, ref_BLK: QuadBlock) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.N.get_south_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockSouth(BoundaryBlock):
    def __init__(self, input_, type_):
        super().__init__(input_, type_)
        self._idx_from_U = slice(0, 4 * self.nx)
        self._state = ConservativeState(input_, self.nx)

    def set(self, ref_BLK: QuadBlock) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.S.get_north_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[2::4] *= -1

class BoundaryBlockEast(BoundaryBlock):
    def __init__(self, input_, type_):
        super().__init__(input_, type_)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(input_, self.ny)

        for j in range(1, self.ny + 1):
            iF = 4 * self.nx * j - 4
            iE = 4 * self.nx * j
            self._idx_from_U[4 * j - 4:4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self, ref_BLK: QuadBlock) -> None:
        if self._type   == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.E.get_west_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1

class BoundaryBlockWest(BoundaryBlock):
    def __init__(self, input_, type_):
        super().__init__(input_, type_)
        self._idx_from_U = np.empty((4*self.ny), dtype=np.int32)
        self._state = ConservativeState(input_, self.ny)

        for j in range(1, self.ny + 1):
            iF = (4 * j - 4) + 4 * (j - 1) * (self.nx - 1)
            iE = (4 * j - 0) + 4 * (j - 1) * (self.ny - 1)
            self._idx_from_U[4 * j - 4: 4 * j] = np.arange(iF, iE, dtype=np.int32)

    def set(self, ref_BLK: QuadBlock) -> None:
        if self._type == 'Outflow':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
        elif self._type == 'None':
            self._state.U = ref_BLK.neighbors.W.get_east_edge()
        elif self._type == 'Reflection':
            self._state.U = ref_BLK.state.U[self._idx_from_U]
            self._state.U[1::4] *= -1
