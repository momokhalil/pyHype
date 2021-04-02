import numpy as np
from pyHype.states import ConservativeState
from pyHype.input_files.input_file_builder import ProblemInput
from pyHype.mesh.mesh_builder import BlockDescription
from pyHype.fvm.methods import FirstOrderUnlimited, SecondOrderLimited
from pyHype.blocks.base import Vertices, Neighbors, BoundaryBlocks
from pyHype.blocks.boundary_blocks import BoundaryBlockEast, BoundaryBlockWest, BoundaryBlockSouth, BoundaryBlockNorth


class Mesh:
    def __init__(self, inputs, mesh_data_):
        self.inputs = inputs
        self.vertices = Vertices(NW=mesh_data_.NW,
                                 NE=mesh_data_.NE,
                                 SW=mesh_data_.SW,
                                 SE=mesh_data_.SE)

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
        meshinputss = self.inputs.meshinputss

        for BLK_data in meshinputss.values():
            self.add(QuadBlock(self.inputs, BLK_data))

        self._number_of_blocks = len(self._blocks)

        for global_nBLK, block in self._blocks.items():
            Neighbor_E_idx = meshinputss.get(block.global_nBLK).NeighborE
            Neighbor_W_idx = meshinputss.get(block.global_nBLK).NeighborW
            Neighbor_N_idx = meshinputss.get(block.global_nBLK).NeighborN
            Neighbor_S_idx = meshinputss.get(block.global_nBLK).NeighborS

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
        self.boundary_blocks = BoundaryBlocks(E=BoundaryBlockEast(self.inputs, block_data.BCTypeE),
                                              W=BoundaryBlockWest(self.inputs, block_data.BCTypeW),
                                              N=BoundaryBlockNorth(self.inputs, block_data.BCTypeN),
                                              S=BoundaryBlockSouth(self.inputs, block_data.BCTypeS))

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
        self.boundary_blocks.E.set(ref_BLK=self)
        self.boundary_blocks.W.set(ref_BLK=self)
        self.boundary_blocks.N.set(ref_BLK=self)
        self.boundary_blocks.S.set(ref_BLK=self)
