import numpy as np
import scipy.sparse as sparse
from abc import ABC, abstractmethod
from pyHype.flux_limiters import van_leer, van_albada
from pyHype.flux.Roe import ROE_FLUX_X, ROE_FLUX_Y
from pyHype.flux.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y


class FiniteVolumeMethod(ABC):
    def __init__(self, inputs, global_nBLK):
        self.inputs = inputs
        self.global_nBLK = global_nBLK
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.Flux_X = np.empty((4 * self.nx * self.ny, 1))
        self.Flux_Y = np.empty((4 * self.nx * self.ny, 1))

        # Set Flux Function
        if self.inputs.flux_function == 'Roe':
            self._flux_function_X = ROE_FLUX_X(self.inputs)
            self._flux_function_Y = ROE_FLUX_Y(self.inputs)

        elif self.inputs.flux_function == 'HLLE':
            self._flux_function_X = HLLE_FLUX_X(self.inputs)
            self._flux_function_Y = HLLE_FLUX_Y(self.inputs)

        elif self.inputs.flux_function == 'HLLL':
            self._flux_function_X = HLLL_FLUX_X(self.inputs)
            self._flux_function_Y = HLLL_FLUX_Y(self.inputs)
        else:
            print('FiniteVolumeMethod: Flux function type not specified. Defaulting to Roe.')
            self._flux_function_X = ROE_FLUX_X(self.inputs)
            self._flux_function_Y = ROE_FLUX_Y(self.inputs)

        # Van Leer limiter
        if self.inputs.flux_limiter == 'van_leer':
            self._flux_limiter = van_leer

        # Van Albada limiter
        elif self.inputs.flux_limiter == 'van_albada':
            self._flux_limiter = van_albada

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


    @abstractmethod
    def get_slope(self, U):
        pass

    @abstractmethod
    def get_limiter(self, U):
        pass

    @abstractmethod
    def reconstruct_state_X(self, U):
        pass

    @abstractmethod
    def reconstruct_state_Y(self, U):
        pass
