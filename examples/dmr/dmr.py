import numpy as np
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import PrimitiveState
from pyhype.solvers.base import ProblemInput
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

_mesh = QuadMeshGenerator(
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

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="HLLL",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK2",
    problem_type="mach_reflection",
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="machref",
    write_every_n_timesteps=20,
    plot_every=1,
    CFL=0.4,
    t_final=0.25,
    realplot=True,
    profile=False,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
    mesh=_mesh,
)

# Create solver
exp = Euler2D(inputs=inputs)

# Solve
exp.solve()
