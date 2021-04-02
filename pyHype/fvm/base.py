import numpy as np
import scipy.sparse as sparse
from abc import ABC, abstractmethod
from pyHype.flux_limiters import van_leer, van_albada
from pyHype.flux_functions.Roe import ROE_FLUX_X, ROE_FLUX_Y
from pyHype.flux_functions.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux_functions.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y


class FiniteVolumeMethod(ABC):
    def __init__(self, input_, global_nBLK):
        self._input = input_
        self._y_index = None
        self._flux_function_X = None
        self._flux_function_Y = None
        self._limiter = None
        self.global_nBLK = global_nBLK
        self.nx = input_.nx
        self.ny = input_.ny
        self.Flux_X = np.empty((4 * self.nx * self.ny, 1))
        self.Flux_Y = np.empty((4 * self.nx * self.ny, 1))
        self._set_flux_function()
        self._set_limiter()

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


    def _set_flux_function(self):

        # Roe flux
        if self._input.flux_function == 'Roe':
            self._flux_function_X = ROE_FLUX_X(self._input)
            self._flux_function_Y = ROE_FLUX_Y(self._input)

        # HLLE flux
        elif self._input.flux_function == 'HLLE':
            self._flux_function_X = HLLE_FLUX_X(self._input)
            self._flux_function_Y = HLLE_FLUX_Y(self._input)

        # HLLL flux
        elif self._input.flux_function == 'HLLL':
            self._flux_function_X = HLLL_FLUX_X(self._input)
            self._flux_function_Y = HLLL_FLUX_Y(self._input)

    def _set_limiter(self):

        # Van Leer limiter
        if self._input.flux_limiter == 'van_leer':
            self._flux_limiter = van_leer

        # Van Albada limiter
        elif self._input.flux_limiter == 'van_albada':
            self._flux_limiter = van_albada

        # No limiter
        elif self._input.flux_limiter == 'none':
            self._flux_limiter = None

    @abstractmethod
    def _get_slope(self, U):
        pass

    @abstractmethod
    def _get_limiter(self, U):
        pass

    @abstractmethod
    def _reconstruct_state_X(self, U):
        pass

    @abstractmethod
    def _reconstruct_state_Y(self, U):
        pass

    def get_flux(self, ref_BLK):
        for i in range(1, self.ny + 1):
            self._reconstruct_state_X(np.vstack((ref_BLK.boundary_blocks.W.state.U[4 * i - 4:4 * i],
                                                 ref_BLK.state.U[4 * self.nx * (i - 1):4 * self.nx * i],
                                                 ref_BLK.boundary_blocks.E.state.U[4 * i - 4:4 * i])))

            flux = self._flux_function_X.get_flux()
            self.Flux_X[4 * self.nx * (i - 1):4 * self.nx * i] = flux[4:] - flux[:-4]

        for i in range(1, self.nx + 1):
            self._reconstruct_state_Y(np.vstack((ref_BLK.boundary_blocks.S.state.U[4 * i - 4:4 * i],
                                                 ref_BLK.state.U[self._y_index + 4 * i],
                                                 ref_BLK.boundary_blocks.N.state.U[4 * i - 4:4 * i])))

            flux = self._flux_function_Y.get_flux()
            self.Flux_Y[4 * self.ny * (i - 1):4 * self.ny * i] = flux[4:] - flux[:-4]

        self.Flux_Y = self._shuffle.dot(self.Flux_Y)
