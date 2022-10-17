from mesh import step_ten_block
from pyhype.solvers import Euler2D
from pyhype.solvers.base import ProblemInput
from pyhype.states import PrimitiveState

mesh = step_ten_block()

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="HLLL",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK2",
    mesh=mesh,
    problem_type="supersonic_flood",
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="super_step",
    write_every_n_timesteps=20,
    plot_every=10,
    CFL=0.3,
    t_final=25.0,
    realplot=True,
    profile=False,
    gamma=1.4,
    rho_inf=1.0,
    a_inf=1.0,
    R=287.0,
    nx=48,
    ny=16,
    nghost=1,
    use_JIT=True,
)


# Create solver
exp = Euler2D(inputs=inputs)

# Solve
exp.solve()
