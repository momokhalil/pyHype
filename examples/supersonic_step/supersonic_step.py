from examples.supersonic_step.mesh import step_ten_block
from pyhype.solvers import Euler2D
from examples.supersonic_step.config import config

if __name__ == "__main__":
    mesh = step_ten_block()
    supersonic_step = Euler2D(config=config, mesh_config=mesh)
    supersonic_step.solve()
