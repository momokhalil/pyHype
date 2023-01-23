from examples.explosion.config import config
from examples.explosion.mesh import mesh_dict

from pyhype.solvers import Euler2D

exp_sim = Euler2D(
    config=config,
    mesh_config=mesh_dict,
)
exp_sim.solve()
