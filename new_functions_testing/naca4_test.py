import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from naca4 import cluster_both, cluster_right, ellipse, trans_finite, converge, plot
from pyHype.mesh.airfoil import NACA4
import numpy as np

farfield = 2
ny = 41
nx = 31
f = 0.0

source = {'ap': 0.5,
          'cp': 1.0,
          'aq': 100.0,
          'cq': 10.0,
          'p': [0, 1],
          'q': [1]
          }

airfoil = NACA4(airfoil='9410',
                angle_start=0,
                angle_end=180,
                aoa=5,
                npt=int(np.floor(ny/2)+1))

# -------------------------------------------------------------
# Mesh

X           = np.zeros((ny, nx))
Y           = np.zeros((ny, nx))

theta       = cluster_both(3 * np.pi / 2, np.pi / 2, ny, factor=1.5)

r           = ellipse(farfield + 1, farfield + 1, theta)

X[:, 0]     = r * np.cos(theta) + 1
X[:, -1]    = np.concatenate((np.flip(airfoil.x_lower), airfoil.x_upper[1:]))
X[-1, :]    = np.linspace(airfoil.x_upper[-1], airfoil.x_upper[-1], nx)
X[0, :]     = np.linspace(airfoil.x_upper[-1], airfoil.x_upper[-1], nx)

Y[:, 0]     = r * np.sin(theta)
Y[:, -1]    = np.concatenate((np.flip(airfoil.y_lower)+f, airfoil.y_upper[1:]+f))
Y[-1, :]    = cluster_right(r[-1], airfoil.y_upper[-1] + f, nx, factor=3, flip=True)
Y[0, :]     = cluster_right(-r[1], airfoil.y_lower[-1] + f, nx, factor=3, flip=True)

X, Y        = trans_finite(X, Y)
Xt, Yt      = X.copy(), Y.copy()

# -------------------------------------------------------------
# Calculate mesh

_eta        = np.linspace(0, 1, nx - 2)
_xi         = np.linspace(0, 1, ny - 2)
eta, xi     = np.meshgrid(_eta, _xi)

X, Y        = converge(X, Y, eta, xi, source)

# Plot
plot(X, Y, Xt, Yt)
plot(X, Y, Xt, Yt, xlim=(-0.25, 1), ylim=(-0.5, 0.5), vert=True)
