from pyhype.solvers import Euler2D
from examples.explosion_multi.config import config
from examples.explosion_multi.mesh import mesh


if __name__ == "__main__":
    exp_sim = Euler2D(
        config=config,
        mesh_config=mesh,
    )
    exp_sim.solve()
