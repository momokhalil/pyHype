import numpy as np

########################################################################################################################
# IMPORTANT !!!!!!!!!!!!!!!!

# This is an experimental development version, these functions will not be used as is in the final version. They are for
# development and debugging purposes. They will be compiled using numba once they are complete, in order to speed up the
# gradient calculation.

# A more general least squares method will be developed and implemented eventually.

########################################################################################################################

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
                                 y1: np.ndarray,
                                 y2: np.ndarray,
                                 y: np.ndarray,
                                 xc: np.float,
                                 yc: np.float,
                                 stencil: list) -> [np.ndarray]:

    xx = ((x1[stencil[0]] - xc) ** 2 +
          (x1[stencil[1]] - xc) ** 2 +
          (x2[stencil[2]] - xc) ** 2 +
          (x2[stencil[3]] - xc) ** 2 +
          (x[stencil[4]]  - xc) ** 2 +
          (x[stencil[5]]  - xc) ** 2 +
          (x[stencil[6]]  - xc) ** 2) / 7

    yy = ((y1[stencil[0]] - yc) ** 2 +
          (y1[stencil[1]] - yc) ** 2 +
          (y2[stencil[2]] - yc) ** 2 +
          (y2[stencil[3]] - yc) ** 2 +
          (y[stencil[4]] - yc) ** 2 +
          (y[stencil[5]] - yc) ** 2 +
          (y[stencil[6]] - yc) ** 2) / 7

    xy = ((x1[stencil[0]] - xc) * (y1[stencil[0]] - yc) +
          (x1[stencil[1]] - xc) * (y1[stencil[1]] - yc) +
          (x2[stencil[2]] - xc) * (y2[stencil[2]] - yc) +
          (x2[stencil[3]] - xc) * (y2[stencil[3]] - yc) +
          (x[stencil[4]] - xc) * (y[stencil[4]] - yc) +
          (x[stencil[5]] - xc) * (y[stencil[5]] - yc) +
          (x[stencil[6]] - xc) * (y[stencil[6]] - yc)) / 7

    return xx, yy, xy


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
    South West Cell Stencil                                 North West Cell Stencil
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


    South East Cell Stencil                                 North East Cell Stencil
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
    xx, yy, xy = _get_spatial_terms_at_corner(x1, x2, x, y1, y2, y, xc, yc, stencil)

    return ux, uy, xx, yy, xy


def least_squares_9_point(Q, QE, QW, QN, QS,
                          x, y, Ex, Ey, Wx, Wy, Nx, Ny, Sx, Sy,
                          nx, ny,
                          stencilSW, stencilNW, stencilSE, stencilNE):

    # Initialize gradient arrays
    dQdx = np.zeros((ny, nx, 4))
    dQdy = np.zeros((ny, nx, 4))

    # ---------------------------------------------------------------------------------------
    # Corner cells

    # -------------------------------
    # South West
    xc, yc = x[0, 0], y[0, 0]
    ux, uy, x2, y2, xy = _corner_cell(Q, QW, QS,
                                      Wx, Wy, Sx, Sy, x, y,
                                      xc, yc,
                                      stencilSW)
    den = y2 * x2 - xy ** 2

    dQdx[0, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[0, 0, :] = (uy * x2 - xy * ux) / den

    # -------------------------------
    # North West
    xc, yc = x[-1, 0], y[-1, 0]
    ux, uy, x2, y2, xy = _corner_cell(Q, QW, QN,
                                      Wx, Wy, Nx, Ny, x, y,
                                      xc, yc,
                                      stencilNW)
    den = y2 * x2 - xy ** 2

    dQdx[-1, 0, :] = (ux * y2 - xy * uy) / den
    dQdy[-1, 0, :] = (uy * x2 - xy * ux) / den

    # -------------------------------
    # South East
    xc, yc = x[0, -1], y[0, -1]
    ux, uy, x2, y2, xy = _corner_cell(Q, QE, QS,
                                      Ex, Ey, Sx, Sy, x, y,
                                      xc, yc,
                                      stencilSE)
    den = y2 * x2 - xy ** 2

    dQdx[0, -1, :] = (ux * y2 - xy * uy) / den
    dQdy[0, -1, :] = (uy * x2 - xy * ux) / den

    # -------------------------------
    # North East
    xc, yc = x[-1, -1], y[-1, -1]
    ux, uy, x2, y2, xy = _corner_cell(Q, QN, QE,
                                      Nx, Ny, Ex, Ey, x, y,
                                      xc, yc,
                                      stencilNE)
    den = y2 * x2 - xy ** 2

    dQdx[-1, -1, :] = (ux * y2 - xy * uy) / den
    dQdy[-1, -1, :] = (uy * x2 - xy * ux) / den

    # ---------------------------------------------------------------------------------------
    # Edges

    # --------------------------------------------------
    # South edge
    for i in range(1, nx - 1):

        ux = Q[0, i - 1, :] * (x[0, i - 1] - xc) + Q[0, i + 1, :] * (x[0, i + 1] - xc)
        uy = Q[0, i - 1, :] * (y[0, i - 1] - yc) + Q[0, i + 1, :] * (y[0, i + 1] - yc)

        xx = (x[0, i - 1] - xc) ** 2 + (x[0, i + 1] - xc) ** 2
        yy = (y[0, i - 1] - yc) ** 2 + (y[0, i + 1] - yc) ** 2
        xy = (x[0, i - 1] - xc) * (y[0, i - 1] - yc) + (x[0, i + 1] - xc) * (y[0, i + 1] - yc)

        for j in range(i-1, i+2):

            ux += QS[0, j, :] * (Sx[0, j] - xc) + Q[1, j, :] * (x[1, j] - xc)
            uy += QS[0, j, :] * (Sy[0, j] - yc) + Q[1, j, :] * (y[1, j] - yc)

            dx_, dy_ = Sy[0, j] - yc, Sx[0, j] - xc
            dx, dy = x[1, j] - xc, y[1, j] - yc

            xx += dx_ * dx_ + dx * dx
            yy += dy_ * dy_ + dy * dy
            xy += dy_ * dx_ + dy * dx

        den = yy * xx - xy ** 2
        dQdx[0, i, :] = (ux * yy - xy * uy) / den
        dQdy[0, i, :] = (uy * xx - xy * ux) / den

    # --------------------------------------------------
    # North edge
    for i in range(1, nx - 1):

        ux = Q[-1, i - 1, :] * (x[-1, i - 1] - xc) + Q[-1, i + 1, :] * (x[-1, i + 1] - xc)
        uy = Q[-1, i - 1, :] * (y[-1, i - 1] - yc) + Q[-1, i + 1, :] * (y[-1, i + 1] - yc)

        xx = (x[-1, i - 1] - xc) ** 2 + (x[-1, i + 1] - xc) ** 2
        yy = (y[-1, i - 1] - yc) ** 2 + (y[-1, i + 1] - yc) ** 2
        xy = (x[-1, i - 1] - xc) * (y[-1, i - 1] - yc) + (x[-1, i + 1] - xc) * (y[-1, i + 1] - yc)

        for j in range(i-1, i+2):

            ux += QN[0, j, :] * (Nx[0, j] - xc) + Q[-2, j, :] * (x[-2, j] - xc)
            uy += QN[0, j, :] * (Ny[0, j] - yc) + Q[-2, j, :] * (y[-2, j] - yc)

            dx_, dy_ = Ny[0, j] - yc, Nx[0, j] - xc
            dx, dy = x[-2, j] - xc, y[-2, j] - yc

            xx += dx_ * dx_ + dx * dx
            yy += dy_ * dy_ + dy * dy
            xy += dy_ * dx_ + dy * dx

        den = yy * xx - xy ** 2
        dQdx[-1, i, :] = (ux * yy - xy * uy) / den
        dQdy[-1, i, :] = (uy * xx - xy * ux) / den

    # --------------------------------------------------
    # West Edge
    for i in range(1, ny - 1):

        ux = Q[i - 1, 0, :] * (x[i - 1, 0] - xc) + Q[i + 1, 0, :] * (x[i + 1, 0] - xc)
        uy = Q[i - 1, 0, :] * (y[i - 1, 0] - yc) + Q[i + 1, 0, :] * (y[i + 1, 0] - yc)

        xx = (x[i - 1, 0] - xc) ** 2 + (x[i + 1, 0] - xc) ** 2
        yy = (y[i - 1, 0] - yc) ** 2 + (y[i + 1, 0] - yc) ** 2
        xy = (x[i - 1, 0] - xc) * (y[i - 1, 0] - yc) + (x[i + 1, 0] - xc) * (y[i + 1, 0] - yc)

        for j in range(i - 1, i + 2):
            ux += QW[j, 0, :] * (Wx[j, 0] - xc) + Q[j, 1, :] * (x[j, 1] - xc)
            uy += QW[j, 0, :] * (Wy[j, 0] - yc) + Q[j, 1, :] * (y[j, 1] - yc)

            dx_, dy_ = Wy[j, 0] - yc, Wx[j, 0] - xc
            dx, dy = x[j, 1] - xc, y[j, 1] - yc

            xx += dx_ * dx_ + dx * dx
            yy += dy_ * dy_ + dy * dy
            xy += dy_ * dx_ + dy * dx

        den = yy * xx - xy ** 2
        dQdx[i, 0, :] = (ux * yy - xy * uy) / den
        dQdy[i, 0, :] = (uy * xx - xy * ux) / den

    # --------------------------------------------------
    # North edge
    for i in range(1, ny - 1):

        ux = Q[i - 1, -1, :] * (x[i - 1, -1] - xc) + Q[i + 1, -1, :] * (x[i + 1, -1] - xc)
        uy = Q[i - 1, -1, :] * (y[i - 1, -1] - yc) + Q[i + 1, -1, :] * (y[i + 1, -1] - yc)

        xx = (x[i - 1, -1] - xc) ** 2 + (x[i + 1, -1] - xc) ** 2
        yy = (y[i - 1, -1] - yc) ** 2 + (y[i + 1, -1] - yc) ** 2
        xy = (x[i - 1, -1] - xc) * (y[i - 1, -1] - yc) + (x[i + 1, -1] - xc) * (y[i + 1, -1] - yc)

        for j in range(i - 1, i + 2):
            ux += QE[j, 0, :] * (Ex[j, 0] - xc) + Q[j, -2, :] * (x[j, -2] - xc)
            uy += QE[j, 0, :] * (Ey[j, 0] - yc) + Q[j, -2, :] * (y[j, -2] - yc)

            dx_, dy_ = Ey[j, 0] - yc, Ex[j, 0] - xc
            dx, dy = x[j, -2] - xc, y[j, -2] - yc

            xx += dx_ * dx_ + dx * dx
            yy += dy_ * dy_ + dy * dy
            xy += dy_ * dx_ + dy * dx

        den = yy * xx - xy ** 2
        dQdx[i, -1, :] = (ux * yy - xy * uy) / den
        dQdy[i, -1, :] = (uy * xx - xy * ux) / den

    # --------------------------------------------------
    # Body
    for i in range(1, ny - 1):
        for k in range(1, nx - 1):

            ux = Q[i, k - 1, :] * (x[i, k - 1] - xc) + Q[i, k + 1, :] * (x[i, k + 1] - xc)
            uy = Q[i, k - 1, :] * (y[i, k - 1] - yc) + Q[i, k + 1, :] * (y[i, k + 1] - yc)

            xx = (x[i, k - 1] - xc) ** 2 + (x[i, k + 1] - xc) ** 2
            yy = (y[i, k - 1] - yc) ** 2 + (y[i, k + 1] - yc) ** 2
            xy = (x[i, k - 1] - xc) * (y[i, k - 1] - yc) + (x[i, k + 1] - xc) * (y[i, k + 1] - yc)

            for j in range(k - 1, k + 2):

                ux += Q[i-1, j, :] * (x[i-1, j] - xc) + Q[i+1, j, :] * (x[i+1, j] - xc)
                uy += Q[i-1, j, :] * (y[i-1, j] - yc) + Q[i+1, j, :] * (y[i+1, j] - yc)

                dx_, dy_ = y[i-1, j] - yc, x[i-1, j] - xc
                dx, dy = x[i+1, j] - xc, y[i+1, j] - yc

                xx += dx_ * dx_ + dx * dx
                yy += dy_ * dy_ + dy * dy
                xy += dy_ * dx_ + dy * dx

            den = yy * xx - xy ** 2
            dQdx[i, k, :] = (ux * yy - xy * uy) / den
            dQdy[i, k, :] = (uy * xx - xy * ux) / den

    return dQdx, dQdy

