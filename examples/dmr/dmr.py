from examples.dmr.config import config
from examples.dmr.mesh import mesh_gen
from pyhype.solvers import Euler2D

if __name__ == "__main__":
    dmr_sim = Euler2D(
        config=config,
        mesh_config=mesh_gen,
    )
    dmr_sim.solve()
