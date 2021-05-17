import numpy as np
from abc import abstractmethod
from pyHype.limiters import limiters
from pyHype.states.states import ConservativeState
from pyHype.flux.Roe import ROE_FLUX_X, ROE_FLUX_Y
from pyHype.flux.HLLE import HLLE_FLUX_X, HLLE_FLUX_Y
from pyHype.flux.HLLL import HLLL_FLUX_X, HLLL_FLUX_Y


class FiniteVolumeMethod:
    def __init__(self, inputs, global_nBLK):
        """
        Solves the euler equations using the finite volume method. Consider a simple 4x4 grid as such

        O---------O--------O---------O
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O---------O--------O---------O
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O---------O--------O---------O
        |         |        |         |
        |         |        |         |
        |         |        |         |
        O---------O--------O---------O

        The matrix structure used for storing solution data in various State classes is a (ny * nx * 4) numpy ndarray
        which has planar dimentions equal to the number of cells in the y and x direction, and a depth of 4. The
        structure looks as follows:

            ______________nx______________
            v                            v

        |>  O---------O---------O---------O---------O ........................ q0 (zeroth state variable)
        |   |         |         |         |         |\
        |   |         |         |         |         |-O ..................... q1 (first state variable)
        |   |         |         |         |         | |\
        |   O---------O---------O---------O---------O |-O .................. q2 (second state variable)
        |   |         |         |         |         |\| |\
        ny  |         |         |         |         |-O |-O ............... q3 (third state variable)
        |   |         |         |         |         | |\| |
        |   O---------O---------O---------O---------O |-O |
        |   |         |         |         |         |\| |\|
        |   |         |         |         |         |-O |-O
        |   |         |         |         |         | |\| |
        |>  O---------O---------O---------O---------O |-O |
            |         |         |         |         |\| |\|
            |         |         |         |         |-O | O
            |         |         |         |         | |\| |
            O---------O---------O---------O---------O |-O |
             \         \         \         \         \| |\|
              O---------O---------O---------O---------O |-O
               \         \         \         \         \| |
                O---------O---------O---------O---------O |
                 \         \         \         \         \|
                  O---------O---------O---------O---------O


        then, cells are constructed as follows:

        O---------O---------O---------O
        |         |         |         |
        |         |         |         |
        |         |         |         |
        O---------O---------O---------O
        |         |         |         |
        |         |    .....x.....    | -- Y+1/2
        |         |    .    |    .    |
        O---------O----x--- C ---x----O--- Y
        |         |    .    |    .    |
        |         |    .....x.....    | -- Y-1/2
        |         |         |         |
        O---------O---------O---------O
                       |    |    |
                   X-1/2    X    X+1/2

        Reduction to 1D problem for each cell:

        x - direction:

        O---------O---------O---------O
        |         |         |         |
        |         |         |         |
        |         |         |         |
        O---------O---------O---------O
        |         |         |         |
        |         |         |         |
      ..|.........|.........|.........|..
      . O----x----O----x--- C ---x----O .
      ..|.........|.........|.........|..
        |         |         |         |
        |         |         |         |
        O---------O---------O---------O

        y - direction:

                          . . .
        O---------O---------O---------O
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        O---------O---------O---------O
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        O---------O-------- C --------O
        |         |       . | .       |
        |         |       . x .       |
        |         |       . | .       |
        O---------O---------O---------O
                          . . .

        """

        self.inputs = inputs
        self.nx = inputs.nx
        self.ny = inputs.ny
        self.global_nBLK = global_nBLK

        self.Flux_X = np.empty((self.ny, self.nx, 4))
        self.Flux_Y = np.empty((self.ny, self.nx, 4))

        self.UL = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)
        self.UR = ConservativeState(self.inputs, nx=self.nx + 1, ny=1)

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
            raise ValueError('FiniteVolumeMethod: Flux function type not specified.')

        # Van Leer limiter
        if self.inputs.flux_limiter == 'van_leer':
            self.flux_limiter = limiters.VanLeer(self.inputs)

        # Van Albada limiter
        elif self.inputs.flux_limiter == 'van_albada':
            self.flux_limiter = limiters.VanAlbada(self.inputs)


    @staticmethod
    def get_row(ref_BLK, index: int) -> np.ndarray:
        return np.vstack((ref_BLK.boundary_blocks.W[index],
                          ref_BLK.row(index),
                          ref_BLK.boundary_blocks.E[index]))

    @staticmethod
    def get_col(ref_BLK, index: int) -> np.ndarray:
        return np.vstack((ref_BLK.boundary_blocks.S[index],
                          ref_BLK.col(index),
                          ref_BLK.boundary_blocks.N[index]))

    @abstractmethod
    def reconstruct_state(self, U):
        pass
