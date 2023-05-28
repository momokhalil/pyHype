from pyhype.solvers import Euler2D
from examples.supersonic_wedge.config import air, config
from examples.supersonic_wedge.mesh import make_supersonic_wedge_mesh

if __name__ == "__main__":
    mesh = make_supersonic_wedge_mesh(fluid=air)
    supersonic_wedge = Euler2D(config=config, mesh_config=mesh)
    supersonic_wedge.solve()
