import numpy as np

########################################################################################################################
# FUNCTIONS FOR 9-POINT STENCIL LEAST SQUARES METHOD FOR CALCULATING SOLUTION GRADIENTS
# DO NOT USE FOR ANY OTHER PURPOSE

def _get_average_value_at_corner(Q1: np.ndarray,
                                 Q2: np.ndarray,
                                 Q: np.ndarray,
                                 x1: np.ndarray,
                                 x2: np.ndarray,
                                 x: np.ndarray,
                                 xc: np.float,
                                 stencil: list) -> np.ndarray:

    _avg = (Q1[stencil[0], :] * (x1[stencil[0]] - xc) +
            Q1[stencil[1], :] * (x1[stencil[1]] - xc) +
            Q2[stencil[2], :] * (x2[stencil[2]] - xc) +
            Q2[stencil[3], :] * (x2[stencil[3]] - xc) +
            Q[stencil[4], :]  * (x[stencil[4]]  - xc) +
            Q[stencil[5], :]  * (x[stencil[5]]  - xc) +
            Q[stencil[6], :]  * (x[stencil[6]]  - xc)) / 7

    return _avg


def _get_spatial_terms_at_corner(x1: np.ndarray,
                                 x2: np.ndarray,
                                 x: np.ndarray,
                                 xc: np.float,
                                 stencil: list) -> np.ndarray:

    xx = ((x1[stencil[0]] - xc) ** 2 +
          (x1[stencil[1]] - xc) ** 2 +
          (x2[stencil[2]] - xc) ** 2 +
          (x2[stencil[3]] - xc) ** 2 +
          (x[stencil[4]]  - xc) ** 2 +
          (x[stencil[5]]  - xc) ** 2 +
          (x[stencil[6]]  - xc) ** 2) / 7

    return xx


def _get_cross_spatial_terms_at_corner(x1: np.ndarray,
                                       x2: np.ndarray,
                                       x: np.ndarray,
                                       y1: np.ndarray,
                                       y2: np.ndarray,
                                       y: np.ndarray,
                                       xc: np.float,
                                       yc: np.float,
                                       stencil: list) -> np.ndarray:

    xy = ((x1[stencil[0]] - xc) * (y1[stencil[0]] - yc) +
          (x1[stencil[1]] - xc) * (y1[stencil[1]] - yc) +
          (x2[stencil[2]] - xc) * (y2[stencil[2]] - yc) +
          (x2[stencil[3]] - xc) * (y2[stencil[3]] - yc) +
          (x[stencil[4]] - xc) * (y[stencil[4]] - yc) +
          (x[stencil[5]] - xc) * (y[stencil[5]] - yc) +
          (x[stencil[6]] - xc) * (y[stencil[6]] - yc)) / 7

    return xy


def _corner_cell(Q: np.ndarray,
                 Q1: np.ndarray,
                 Q2: np.ndarray,
                 x1: np.ndarray,
                 y1: np.ndarray,
                 x2: np.ndarray,
                 y2: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 xc: np.float,
                 yc: np.float,
                 stencil: list) -> [np.ndarray]:

    """
                                                            ...........................
          0     O----------O----------O----------O          .           x----------x--.-------O----------O
          |     |          |          |          |          .                         .
          |     |          |          |          |          .     x     C----------x--.-------O----------O
          |     |          |          |          |          .     |     |          |  .       |          |
          0     O----------O----------O----------O          .     |     |          |  .       |          |
          |     |          |          |          |          .     |     |          |  .       |          |
     ......................|...       |          |          .     x     x----------x--.-------O----------O
     .    |     |          |  .       |          |          .     |     |          |  .       |          |
     .    x     x----------x--.-------O----------O          ...........................       |          |
     .    |     |          |  .       |          |                |     |          |          |          |
     .    |     |          |  .       |          |                O     O----------O----------O----------O
     .    |     |          |  .       |          |                |     |          |          |          |
     .    x     C----------x--.-------O----------O                |     |          |          |          |
     .                        .                                   |     |          |          |          |
     .          x----------x--.-------O----------O                O     O----------O----------O----------O
     ..........................


                O----------O----------O----------O     0                                   ..........................
                |          |          |          |     |                O----------O-------.--x----------x          .
                |          |          |          |     |                                   .                        .
                |          |          |          |     |                O----------O-------.--x----------C     x    .
                O----------O----------O----------O     0                |          |       .  |          |     |    .
                |          |          |          |     |                |          |       .  |          |     |    .
                |          |       ....................|.....           |          |       .  |          |     |    .
                |          |       .  |          |     |    .           O----------O-------.--x----------x     x    .
                O----------O-------.--x----------x     x    .           |          |       .  |          |     |    .
                |          |       .  |          |     |    .           |          |       ..........................
                |          |       .  |          |     |    .           |          |          |          |     |
                |          |       .  |          |     |    .           O----------O----------O----------O     O
                O----------O-------.--x----------C     x    .           |          |          |          |     |
                                   .                        .           |          |          |          |     |
                O----------O-------.--x----------x          .           |          |          |          |     |
                                   ..........................           O----------O----------O----------O     O


    """


    # Get averaged solution values in each coordinate direction
    ux = _get_average_value_at_corner(Q1, Q2, Q, x1, x2, x, xc, stencil)
    uy = _get_average_value_at_corner(Q1, Q2, Q, y1, y2, y, yc, stencil)

    # Get spatial terms
    x2 = _get_spatial_terms_at_corner(x1, x2, x, xc, stencil)
    y2 = _get_spatial_terms_at_corner(y1, y2, y, yc, stencil)
    xy = _get_cross_spatial_terms_at_corner(x1, x2, x, y1, y2, y, xc, yc, stencil)

    return ux, uy, x2, y2, xy


def least_squares_nearest_neighbor(Q, QE, QW, QN, QS,
                                   x, y, Ex, Ey, Wx, Wy, Nx, Ny, Sx, Sy,
                                   nx, ny):

    dQdx = np.zeros((ny, nx, 4))
    dQdy = np.zeros((ny, nx, 4))

    # Corner cells

    # South West
    xc, yc = x[0, 0], y[0, 0]
    stencil = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]]
    ux, uy, x2, y2, xy = _corner_cell(Q, QW, QS, Wx, Wy, Sx, Sy, x, y, xc, yc, stencil)

    den = y2 * x2 - xy ** 2

    dQdx[0, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[0, 0, :] = (uy * x2 - xy * ux) / den

    # North West
    xc, yc = x[-1, 0], y[-1, 0]
    stencil = [[0, 0], [0, 1], [-2, 0], [-1, 0], [-2, 0], [-2, 1], [-1, 1]]
    ux, uy, x2, y2, xy = _corner_cell(Q, QW, QN, Wx, Wy, Nx, Ny, x, y, xc, yc, stencil)

    den = y2 * x2 - xy ** 2

    dQdx[-1, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[-1, 0, :] = (uy * x2 - xy * ux) / den

    # South East
    xc, yc = x[0, -1], y[0, -1]
    stencil = [[0, -1], [0, -2], [0, 0], [1, 0], [0, -1], [1, -1], [1, -2]]
    ux, uy, x2, y2, xy = _corner_cell(Q, QE, QS, Ex, Ey, Sx, Sy, x, y, xc, yc, stencil)

    den = y2 * x2 - xy ** 2

    dQdx[0, -1, :] = (ux * y2 - xy * uy) / den
    dQdy[0, -1, :] = (uy * x2 - xy * ux) / den

    # North East
    xc, yc = x[-1, -1], y[-1, -1]
    stencil = [[0, -1], [0, -2], [-1, 0], [-2, 0], [-1, -2], [1, -1], [1, -2]]
    ux, uy, x2, y2, xy = _corner_cell(Q, QN, QE, Nx, Ny, Ex, Ey, x, y, xc, yc, stencil)

    den = y2 * x2 - xy ** 2

    dQdx[-1, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[-1, 0, :] = (uy * x2 - xy * ux) / den

    return dQdx, dQdy
