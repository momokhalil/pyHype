from pyhype.solvers import Euler2D

from examples.explosion.config import config
from examples.explosion.mesh import mesh_dict
from pyhype.fvm import FiniteVolumeMethodFactory
from pyhype.flux import FluxFunctionFactory
from pyhype.limiters import SlopeLimiterFactory
from pyhype.gradients import GradientFactory

flux = FluxFunctionFactory.get(config=config)
gradient = GradientFactory.get(config=config)
limiter = SlopeLimiterFactory.get(config=config)
fvm = FiniteVolumeMethodFactory.get(
    config=config,
    flux=flux,
    limiter=limiter,
    gradient=gradient,
)

exp = Euler2D(config=config, mesh_config=mesh_dict, fvm=fvm)
exp.solve()
