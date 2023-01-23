import numpy as np
from pyhype.mesh.base import QuadMeshGenerator

k = 1
a = 2 / np.sqrt(3)
d = np.tan(30 * np.pi / 180)

_left_x = [0, 0]
_left_y = [0, a]
_right_x = [4 * k, 4 * k]
_right_y = [3 * d, a + 3 * d]
_x = [0, k, 2 * k, 3 * k, 4 * k]
_top_y = [a, a, a + d, a + 2 * d, a + 3 * d]
_bot_y = [0, 0, d, 2 * d, 3 * d]

BCS = ["OutletDirichlet", "Slipwall", "Slipwall", "Slipwall"]

mesh_gen = QuadMeshGenerator(
    nx_blk=4,
    ny_blk=1,
    BCE=["OutletDirichlet"],
    BCW=["OutletDirichlet"],
    BCN=["OutletDirichlet"],
    BCS=BCS,
    top_x=_x,
    bot_x=_x,
    top_y=_top_y,
    bot_y=_bot_y,
    left_x=_left_x,
    right_x=_right_x,
    left_y=_left_y,
    right_y=_right_y,
)
