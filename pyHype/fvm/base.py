import numpy as np
import scipy.sparse as sparse
from abc import ABC, abstractmethod
from pyHype.flux.Roe import ROE_FLUX_X, ROE_FLUX_Y
from pyHype.flux.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y
from pyHype.limiters import limiters
from pyHype.states.states import ConservativeState


class FiniteVolumeMethod:
    def __init__(self, inputs, global_nBLK):
        """
        Solves the euler equations using the finite volume method. Consider a simple 4x4 grid as such

        O ------- 0 ------ 0 ------- 0
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O ------- 0 ------ 0 ------- 0
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O ------- 0 ------ 0 ------- 0
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O ------- 0 ------ 0 ------- 0

        then, cells are constructed as follows:

        O ------- 0 ------- 0 ------- 0
        |         |         |         |
        |         |         |         |
        |         |         |         |
        0 ------- 0 ------- 0 ------- 0
        |         |         |         |
        |         |    .....x.....    | -- Y+1/2
        |         |    .    |    .    |
        0 ------- 0 ---x--- C ---x--- 0 -- Y
        |         |    .    |    .    |
        |         |    .....x.....    | -- Y-1/2
        |         |         |         |
        0 ------- 0 ------- 0 ------- 0
                       |    |    |
                   X-1/2    X    X+1/2

        Reduction to 1D problem for each cell:

        x - direction:

        O ------- 0 ------- 0 ------- 0
        |         |         |         |
        |         |         |         |
        |         |         |         |
        0 ------- 0 ------- 0 ------- 0
        |         |         |         |
        |         |         |         |
      ..|.........|.........|.........|..
      . 0 ---x--- 0 ---x--- C ---x--- 0 .
      ..|.........|.........|.........|..
        |         |         |         |
        |         |         |         |
        0 ------- 0 ------- 0 ------- 0

        y - direction:

                          . . .
        O ------- 0 ------- 0 ------- 0
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        0 ------- 0 ------- 0 ------- 0
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        0 ------- 0 ------- C ------- 0
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        0 ------- 0 ------- 0 ------- 0
                          . . .

        """

        self.inputs = inputs
        self.global_nBLK = global_nBLK
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.Flux_X = np.empty((4 * self.nx * self.ny, 1))
        self.Flux_Y = np.empty((4 * self.nx * self.ny, 1))
        self.UL = ConservativeState(self.inputs, self.nx + 1)
        self.UR = ConservativeState(self.inputs, self.nx + 1)

        # Set Flux Function
        if self.inputs.flux_function == 'Roe':
            self.flux_function_X = ROE_FLUX_X(self.inputs)
            self.flux_function_Y = ROE_FLUX_Y(self.inputs)

        elif self.inputs.flux_function == 'HLLE':
            self.flux_function_X = HLLE_FLUX_X(self.inputs)
            self.flux_function_Y = HLLE_FLUX_Y(self.inputs)

        elif self.inputs.flux_function == 'HLLL':
            self.flux_function_X = HLLL_FLUX_X(self.inputs)
            self.flux_function_Y = HLLL_FLUX_Y(self.inputs)
        else:
            print('FiniteVolumeMethod: Flux function type not specified. Defaulting to Roe.')
            self.flux_function_X = ROE_FLUX_X(self.inputs)
            self.flux_function_Y = ROE_FLUX_Y(self.inputs)

        # Van Leer limiter
        if self.inputs.flux_limiter == 'van_leer':
            self.flux_limiter = limiters.VanLeer(self.inputs)

        # Van Albada limiter
        elif self.inputs.flux_limiter == 'van_albada':
            self.flux_limiter = limiters.VanAlbada(self.inputs)

        # Construct indices to access column-wise elements on the mesh
        self._y_index = np.ones((4 * self.ny), dtype=np.int32)

        for i in range(1, self.ny + 1):
            self._y_index[4 * i - 4:4 * i] = np.arange(4 * self.nx * (i - 1) - 4, 4 * self.nx * (i - 1))

        # Construct shuffle matrix
        i = np.arange(0, 4 * self.nx * self.ny)
        j = np.arange(0, 4 * self.nx * self.ny)
        m = np.arange(0, 4 * self.ny)

        for k in range(0, self.ny):
            m[4 * k:4 * (k + 1)] = np.arange(4 * k * self.nx, 4 * (k * self.nx + 1))

        for k in range(0, self.nx):
            j[4 * k * self.ny:4 * (k + 1) * self.ny] = m + 4 * k

        self._shuffle = sparse.coo_matrix((np.ones((4 * self.nx * self.ny)), (i, j)))

    def reconstruct_state_X(self, U):
        pass

    def reconstruct_state_Y(self, U):
        pass
