import numpy as np
from block import QuadBlock
from abc import ABC, abstractmethod
from scipy.sparse import bsr_matrix as bsrmat
from flux_functions import ROE_FLUX_X, ROE_FLUX_Y, HLLE_FLUX_X, HLLE_FLUX_Y, HLLL_FLUX_X, HLLL_FLUX_Y
from flux_limiters import van_leer, van_albada


class FiniteVolumeMethod(ABC):
    def __init__(self, input_, global_nBLK):
        self._input = input_
        self._y_index = None
        self._flux_function_X = None
        self._flux_function_Y = None
        self._limiter = None
        self.global_nBLK = global_nBLK
        self.nx = input_.get('mesh_inputs').get('nx')
        self.ny = input_.get('mesh_inputs').get('ny')
        self.Flux_X = np.empty((4 * self.nx * self.ny, 1))
        self.Flux_Y = np.empty((4 * self.nx * self.ny, 1))
        self._set_flux_function()
        self._set_limiter()

        self._y_index = np.ones((4 * self.ny))
        for i in range(1, self.ny + 1):
            self._y_index[4 * i - 4:4 * i] = np.arange(4 * self.nx * (i - 1) - 4, 4 * self.nx * (i - 1))

        i, j    = np.arange(0, 4 * self.nx * self.ny), np.arange(0, 4 * self.nx * self.ny)
        eye     = np.ones((4 * self.nx * self.ny))
        jj      = np.arange(0, 4 * self.ny)

        for k in range(0, self.ny):
            jj[4 * k:4 * (k + 1)] = np.arange(4 * k * self.nx, 4 * (k * self.nx + 1))

        for k in range(0, self.nx):
            j[4 * k * self.ny:4 * (k + 1) * self.ny] = jj + 4 * k

        self._shuffle = bsrmat((eye, (i, j)))

    def _set_flux_function(self):
        if 'flux_function' not in self._input.keys():
            raise ValueError('flux_function not defined in input file')
        else:
            flux_function = self._input.get('flux_function')
            if flux_function == 'Roe':
                self._flux_function_X = ROE_FLUX_X
                self._flux_function_Y = ROE_FLUX_Y
            elif flux_function == 'HLLE':
                self._flux_function_X = HLLE_FLUX_X
                self._flux_function_Y = HLLE_FLUX_Y
            elif flux_function == 'HLLL':
                self._flux_function_X = HLLL_FLUX_X
                self._flux_function_Y = HLLL_FLUX_Y

    def _set_limiter(self):
        if 'flux_limiter' not in self._input.keys():
            raise ValueError('flux_limiter not defined in input file')
        else:
            flux_limiter = self._input.get('flux_limiter')
            if flux_limiter == 'van_leer':
                self._flux_limiter = van_leer
            elif flux_limiter == 'van_albada':
                self._flux_limiter = van_albada
            elif flux_limiter == 'none':
                self._flux_limiter = None

    @abstractmethod
    def _get_slope(self, U): pass

    @abstractmethod
    def _get_limiter(self, U): pass

    @abstractmethod
    def _reconstruct_state_X(self, U): pass

    @abstractmethod
    def _reconstruct_state_Y(self, U): pass

    def get_flux(self, ref_BLK):
        y_index = self._y_index
        for i in range(1, self.ny + 1):
            self._reconstruct_state_X(np.vstack((ref_BLK.boundary_blocks.W.state.U[4 * i - 4:4 * i],
                                                 ref_BLK.state.U[4 * self.nx * (i - 1):4 * self.nx * i],
                                                 ref_BLK.boundary_blocks.E.state.U[4 * i - 4:4 * i])))

            flux = self._flux_function_X.get_flux()
            self.Flux_X[4 * self.nx * (i - 1):4 * self.nx * i] = flux[4:] - flux[:-4]

        for i in range(1, self.nx + 1):
            y_index += 4
            self._reconstruct_state_Y(np.vstack((ref_BLK.boundary_blocks.S.state.U[4 * i - 4:4 * i],
                                                 ref_BLK.state.U[y_index],
                                                 ref_BLK.boundary_blocks.N.state.U[4 * i - 4:4 * i])))

            flux = self._flux_function_Y.get_flux()
            self.Flux_Y[4 * self.ny * (i - 1):4 * self.ny * i] = flux[4:] - flux[:-4]

        self.Flux_Y = self._shuffle.multiply(self.Flux_Y)


class FirstOrderUnlimited(FiniteVolumeMethod):
    def __init__(self, input_, global_nBLK):
        super().__init__(input_, global_nBLK)

    def _get_slope(self, U): pass

    def _get_limiter(self, U): pass

    def _reconstruct_state_X(self, U):
        self._flux_function_X.set_left_state(U[:-4])
        self._flux_function_X.set_right_state(U[4:])

    def _reconstruct_state_Y(self, U):
        self._flux_function_Y.set_left_state(U[:-4])
        self._flux_function_Y.set_right_state(U[4:])


class SecondOrderLimited(FiniteVolumeMethod):
    def __init__(self, input_, global_nBLK):
        super().__init__(input_, global_nBLK)
        self._slope = None

    def _get_slope(self, U):
        slope = (U[8:] - U[4:-4]) / (U[4:-4] - U[:-8] + 1e-8)
        return slope * (slope > 0)

    def _get_limiter(self, U):
        slope = self._get_slope(U)
        return 0.5 * (self._flux_limiter(slope)) / (slope + 1)

    def _reconstruct_state_X(self, U):
        limited_state   = self._get_limiter(U) * (U[8:] - U[:-8])
        left, right     = U[:-4], U[4:]
        left[4:]        += limited_state
        right[:-4]      -= limited_state

        self._flux_function_X.set_left_state(left)
        self._flux_function_X.set_right_state(right)

    def _reconstruct_state_Y(self, U):
        limited_state   = self._get_limiter(U) * (U[8:] - U[:-8])
        left, right     = U[:-4], U[4:]
        left[4:]        += limited_state
        right[:-4]      -= limited_state

        self._flux_function_Y.set_left_state(left)
        self._flux_function_Y.set_right_state(right)
