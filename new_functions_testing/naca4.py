import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

np.set_printoptions(precision=3)

def residual(xo, xn, yo, yn):
    return np.linalg.norm(np.sqrt((xo - xn) ** 2 + (yo - yn) ** 2), 1)

def gauss_seidel(alpha: np.ndarray,
                 beta: np.ndarray,
                 gamma: np.ndarray,
                 x: np.ndarray,
                 dxi: float,
                 deta: float,
                 ) -> [np.ndarray]:

    # Gauss seidel update for the laplace equation in xi and eta
    xij = alpha * (x[2:, 1:-1] + x[:-2, 1:-1]) / (dxi ** 2) \
        - 0.5 * beta * (x[2:, 2:] - x[:-2, 2:] - x[2:, :-2] + x[:-2, :-2]) / (dxi * deta) \
        + gamma * (x[1:-1, 2:] + x[1:-1, :-2]) / (deta ** 2)

    return xij

def compute_solution_update(x: np.ndarray,
                            y: np.ndarray,
                            source: dict,
                            dxi: float,
                            deta: float,
                            eta: np.ndarray,
                            xi: np.ndarray,
                            ) -> [np.ndarray]:

    # Grid Derivatives
    xeta    = dxdeta(x, deta)       # dx/deta
    xxi     = dxdxi(x, dxi)         # dx/deta
    yeta    = dydeta(y, deta)       # dy/dxi
    yxi     = dydxi(y, dxi)         # dy/dxi

    # Grid metrics
    alpha   = get_alpha(xeta, yeta)
    beta    = get_beta(xeta, yeta, xxi, yxi)
    gamma   = get_gamma(xxi, yxi)

    # x coordinate update with Laplace only
    xij     = gauss_seidel(alpha, beta, gamma, x, dxi, deta)
    # y coordinate update with Laplace only
    yij     = gauss_seidel(alpha, beta, gamma, y, dxi, deta)

    # Jacobian squares
    J2      = Jacobian(xeta, yeta, xxi, yxi) ** 2
    # Update multiplier
    K       = 0.5 * (dxi ** 2 * deta ** 2) / (alpha * deta ** 2 + gamma * dxi ** 2 + 1e-9)
    # x and y direction control functions
    P, Q    = control_functions(eta, xi, source)
    # x and y direction source terms
    xp      = J2 * (xxi * P + xeta * Q)
    yp      = J2 * (yxi * P + yeta * Q)

    # x and y coordinate updates corrected with source terms
    xij     = (xij + xp) * K
    yij     = (yij + yp) * K

    return xij, yij

def control_functions(eta, xi, source):

    P = sum([cluster(source['ap'], source['cp'], xi, pt) for pt in source['p']])
    Q = sum([cluster(source['aq'], source['cq'], eta, pt) for pt in source['q']])

    return P, Q

def cluster(a, c, x, p):
    return - a * np.sign(x - p) * np.exp(-c * np.abs(x - p))

def trans_finite(x, y):

    ny, nx = x.shape

    _im = np.linspace(1 / ny, (ny - 1) / ny, ny - 2)
    _jm = np.linspace(1 / nx, (nx - 1) / nx, nx - 2)

    jmm, imm = np.meshgrid(_jm, _im)

    _mi = np.linspace((ny - 1) / ny, 1 / ny, ny - 2)
    _mj = np.linspace((nx - 1) / nx, 1 / nx, nx - 2)

    mmj, mmi = np.meshgrid(_mj, _mi)

    x[1:-1, 1:-1] = mmi * x[0, 1:-1]            \
                  + imm * x[-1, 1:-1]           \
                  + mmj * x[1:-1, 0, None]      \
                  + jmm * x[1:-1, -1, None]     \
                  - mmi * mmj * x[0, 0]         \
                  - mmi * jmm * x[0, -1]        \
                  - imm * mmj * x[-1, 0]        \
                  - imm * jmm * x[-1, -1]

    y[1:-1, 1:-1] = mmi * y[0, 1:-1] \
                  + imm * y[-1, 1:-1] \
                  + mmj * y[1:-1, 0, None] \
                  + jmm * y[1:-1, -1, None] \
                  - mmi * mmj * y[0, 0] \
                  - mmi * jmm * y[0, -1] \
                  - imm * mmj * y[-1, 0] \
                  - imm * jmm * y[-1, -1]

    return x, y

def get_alpha(xeta: np.ndarray,
              yeta: np.ndarray
              ) -> [np.ndarray]:
    return xeta ** 2 + yeta ** 2

def get_gamma(xxi: np.ndarray,
              yxi: np.ndarray
              ) -> [np.ndarray]:
    return xxi ** 2 + yxi ** 2

def get_beta(xeta: np.ndarray,
             yeta: np.ndarray,
             xxi: np.ndarray,
             yxi: np.ndarray
             ) -> [np.ndarray]:
    return xxi * xeta + yxi * yeta

def Jacobian(xeta: np.ndarray,
             yeta: np.ndarray,
             xxi: np.ndarray,
             yxi: np.ndarray
             ) -> [np.ndarray]:
    return xxi * yeta - xeta * yxi

def dxdxi(x: np.ndarray,
          dxi: float
          ) -> [np.ndarray]:
    return 0.5 * (x[2:, 1:-1] - x[:-2, 1:-1]) / dxi

def dydxi(y: np.ndarray,
          dxi: float
          ) -> [np.ndarray]:
    return 0.5 * (y[2:, 1:-1] - y[:-2, 1:-1]) / dxi

def dxdeta(x: np.ndarray,
           deta: float
           ) -> [np.ndarray]:
    return 0.5 * (x[1:-1, 2:] - x[1:-1, :-2]) / deta

def dydeta(y: np.ndarray,
           deta: float
           ) -> [np.ndarray]:
    return 0.5 * (y[1:-1, 2:] - y[1:-1, :-2]) / deta

def converge(x: np.ndarray,
             y: np.ndarray,
             eta: np.ndarray,
             xi: np.ndarray,
             source: dict,
             outflow: bool = False,
             res_crit: float = 0.00001,
             w: float = 0.5):

    res = 1000

    ny, nx = x.shape

    dxi = 1 / (ny - 1)
    deta = 1 / (nx - 1)

    while res > res_crit:

        x_new, y_new = compute_solution_update(x, y, source, dxi, deta, eta, xi)

        res = residual(x[1:-1, 1:-1], x_new, y[1:-1, 1:-1], y_new)

        x[1:-1, 1:-1] = w * x_new + (1 - w) * x[1:-1, 1:-1]
        y[1:-1, 1:-1] = w * y_new + (1 - w) * y[1:-1, 1:-1]

        if outflow:
            y[0, 1:-1] = y[1, 1:-1]
            y[-1, 1:-1] = y[-2, 1:-1]

    return x, y

def cluster_right(start, end, n, factor: float = 2.0, flip: bool = False):
    length = end - start
    _x = np.linspace(0, 1, n)
    _s = np.tanh(factor * (1 - _x)) / np.tanh(factor)
    _s *= length
    _s += start
    return np.flip(_s) if flip else _s

def cluster_left(start, end, n, factor: float = 2.0, flip: bool = False):
    length = end - start
    _x = np.linspace(0, 1, n)
    _s = 1 - np.tanh(factor * (1 - _x)) / np.tanh(factor)
    _s *= length
    _s += start
    return np.flip(_s) if flip else _s

def cluster_both(start, end, n, factor: float = 2.0, flip: bool = False):
    length = end - start
    _x = np.linspace(-1, 1, n)
    _s = np.tanh(factor * _x) / np.tanh(factor)
    _s += 1
    _s /= 2
    _s *= length
    _s += start
    return np.flip(_s) if flip else _s

def ellipse(a, b, theta):
    return a * b / (np.sqrt((b * np.sin(theta)) ** 2 + (a * np.cos(theta)) ** 2))

def plot(X: np.ndarray,
         Y: np.ndarray,
         Xt: np.ndarray,
         Yt: np.ndarray,
         xlim: [float, int] = None,
         ylim: [float, int] = None,
         vert: bool = False):

    if vert:
        fig, ax = plt.subplots(2)
    else:
        fig, ax = plt.subplots(1, 2)

    ax[0].scatter(X, Y, color='black', s=0.0)
    segs1 = np.stack((X, Y), axis=2)
    segs2 = segs1.transpose((1, 0, 2))
    ax[0].add_collection(LineCollection(segs1, colors='black', linewidths=1))
    ax[0].add_collection(LineCollection(segs2, colors='black', linewidths=1))
    ax[0].set_aspect('equal', adjustable='box')
    ax[0].set_title('Optimized Mesh (Poisson)')

    ax[1].scatter(Xt, Yt, color='black', s=0.0)
    segs1 = np.stack((Xt, Yt), axis=2)
    segs2 = segs1.transpose((1, 0, 2))
    ax[1].add_collection(LineCollection(segs1, colors='black', linewidths=1))
    ax[1].add_collection(LineCollection(segs2, colors='black', linewidths=1))
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_title('Trans-finite interpolation mesh')

    if xlim:
        for axs in ax:
            axs.set_xlim(xlim)

    if ylim:
        for axs in ax:
            axs.set_ylim((-0.5, 0.5))

    plt.show()
    plt.close()

