import numpy as np
from pyHype.states import ConservativeState, PrimitiveState
from pyHype.input.input_file_builder import ProblemInput

class GradientSolver:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs


    def least_squares_nearest_neighbor(self, refBLK):

        bdr = refBLK.boundary_blocks

        dQdx, dQdy = _least_squares_nearest_neighbor(refBLK.state.Q,
                                                     bdr.E.state.Q, bdr.W.state.Q, bdr.N.state.Q, bdr.S.state.Q,
                                                     refBLK.mesh.x, refBLK.mesh.y,
                                                     bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                                     bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                                     self.inputs.nx, self.inputs.ny)
        return dQdx, dQdy


def _south_west_cell(Q, QW, QS, Sx, Sy, Wx, Wy, x, y):

    xc, yc = x[0, 0], y[0, 0]

    ux = (QS[0, 0, :] * (Sx[0, 0] - xc) +
          QS[0, 1, :] * (Sx[0, 1] - xc) +
          QW[0, 0, :] * (Wx[0, 0] - xc) +
          QW[1, 0, :] * (Wx[1, 0] - xc) +
          Q[0, 1, :] * (x[0, 1] - xc) +
          Q[1, 0, :] * (x[1, 0] - xc) +
          Q[1, 1, :] * (x[1, 1] - xc)) / 7

    uy = (QS[0, 0, :] * (Sy[0, 0] - yc) +
          QS[0, 1, :] * (Sy[0, 1] - yc) +
          QW[0, 0, :] * (Wy[0, 0] - yc) +
          QW[1, 0, :] * (Wy[1, 0] - yc) +
          Q[0, 1, :] * (y[0, 1] - yc) +
          Q[1, 0, :] * (y[1, 0] - yc) +
          Q[1, 1, :] * (y[1, 1] - yc)) / 7

    x2 = ((Sx[0, 0] - xc) ** 2 +
          (Sx[0, 1] - xc) ** 2 +
          (Wx[0, 0] - xc) ** 2 +
          (Wx[1, 0] - xc) ** 2 +
          (x[0, 1] - xc) ** 2 +
          (x[1, 0] - xc) ** 2 +
          (x[1, 1] - xc) ** 2) / 7

    y2 = ((Sy[0, 0] - yc) ** 2 +
          (Sy[0, 1] - yc) ** 2 +
          (Wy[0, 0] - yc) ** 2 +
          (Wy[1, 0] - yc) ** 2 +
          (y[0, 1] - yc) ** 2 +
          (y[1, 0] - yc) ** 2 +
          (y[1, 1] - yc) ** 2) / 7

    xy = ((Sy[0, 0] - yc) * (Sx[0, 0] - xc) +
          (Sy[0, 1] - yc) * (Sx[0, 1] - xc) +
          (Wy[0, 0] - yc) * (Wx[0, 0] - xc) +
          (Wy[1, 0] - yc) * (Wx[1, 0] - xc) +
          (y[0, 1] - yc) * (x[0, 1] - xc) +
          (y[1, 0] - yc) * (x[1, 0] - xc) +
          (y[1, 1] - yc) * (x[1, 1] - xc)) / 7

    return ux, uy, x2, y2, xy


def _south_east_cell(Q, QE, QS, Sx, Sy, Ex, Ey, x, y):

    xc, yc = x[0, -1], y[0, -1]

    ux = (QS[0, -1, :] * (Sx[0, -1] - xc) +
          QS[0, -2, :] * (Sx[0, -2] - xc) +
          QE[0, 0, :] * (Ex[0, 0] - xc) +
          QE[1, 0, :] * (Ex[1, 0] - xc) +
          Q[0, -1, :] * (x[0, -1] - xc) +
          Q[1, -1, :] * (x[1, -1] - xc) +
          Q[1, -2, :] * (x[1, -2] - xc)) / 7

    uy = (QS[0, -1, :] * (Sy[0, -1] - yc) +
          QS[0, -2, :] * (Sy[0, -2] - yc) +
          QE[0, 0, :] * (Ey[0, 0] - yc) +
          QE[2, 0, :] * (Ey[2, 0] - yc) +
          Q[0, -1, :] * (y[0, -1] - yc) +
          Q[1, -1, :] * (y[1, -1] - yc) +
          Q[1, -2, :] * (y[1, -2] - yc)) / 7

    x2 = ((Sx[0, -1] - xc) ** 2 +
          (Sx[0, -2] - xc) ** 2 +
          (Ex[0, 0] - xc) ** 2 +
          (Ex[1, 0] - xc) ** 2 +
          (x[0, -1] - xc) ** 2 +
          (x[1, -1] - xc) ** 2 +
          (x[1, -2] - xc) ** 2) / 7

    y2 = ((Sy[0, -1] - yc) ** 2 +
          (Sy[0, -2] - yc) ** 2 +
          (Ey[0, 0] - yc) ** 2 +
          (Ey[1, 0] - yc) ** 2 +
          (y[0, -1] - yc) ** 2 +
          (y[1, -1] - yc) ** 2 +
          (y[1, -2] - yc) ** 2) / 7

    xy = ((Sy[0, -1] - yc) * (Sx[0, -1] - xc) +
          (Sy[0, -2] - yc) * (Sx[0, -2] - xc) +
          (Ey[0, 0] - yc) * (Ex[0, 0] - xc) +
          (Ey[1, 0] - yc) * (Ex[1, 0] - xc) +
          (y[0, -1] - yc) * (x[0, -1] - xc) +
          (y[1, -1] - yc) * (x[1, -1] - xc) +
          (y[1, -2] - yc) * (x[1, -2] - xc)) / 7

    return ux, uy, x2, y2, xy


def _least_squares_nearest_neighbor(Q,
                                    QE, QW, QN, QS,
                                    x, y,
                                    Ex, Ey, Wx, Wy, Nx, Ny, Sx, Sy,
                                    nx, ny):

    dQdx = np.zeros((ny, nx, 4))
    dQdy = np.zeros((ny, nx, 4))

    # Corner cells

    # South West -------------------------------------------------------------------------------------------------------
    ux, uy, x2, y2, xy = _south_west_cell(Q, QW, QS, Sx, Sy, Wx, Wy, x, y)

    den = y2 * x2 - xy ** 2

    dQdx[0, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[0, 0, :] = (uy * x2 - xy * ux) / den

    # South East -------------------------------------------------------------------------------------------------------
    ux, uy, x2, y2, xy = _south_east_cell(Q, QE, QS, Sx, Sy, Ex, Ey, x, y)

    den = y2 * x2 - xy ** 2

    dQdx[0, -1, :] = (ux * y2 - xy * uy) / den
    dQdy[0, -1, :] = (uy * x2 - xy * ux) / den





    return dQdx, dQdy
