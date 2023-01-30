from pyhype.fluids import Air
from pyhype.solver_config import SolverConfig
from pyhype.states import ConservativeState
from examples.explosion.initial_condition import ExplosionInitialCondition

air = Air(a_inf=343.0, rho_inf=1.0)

config = SolverConfig(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="Roe",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK4",
    initial_condition=ExplosionInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=ConservativeState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="nozzle",
    write_every_n_timesteps=40,
    plot_every=10,
    CFL=0.7,
    t_final=0.07,
    realplot=False,
    profile=True,
    fluid=air,
    nx=50,
    ny=100,
    nghost=1,
    use_JIT=True,
)
