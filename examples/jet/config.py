from pyhype.fluids import Air
from pyhype.states import PrimitiveState
from pyhype.solver_config import SolverConfig
from examples.jet.initial_condition import SubsonicJetInitialCondition

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

# Solver settings
config = SolverConfig(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="HLLL",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    initial_condition=SubsonicJetInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="kvi",
    write_every_n_timesteps=20,
    plot_every=5,
    CFL=0.4,
    t_final=25.0,
    realplot=True,
    profile=False,
    fluid=air,
    nx=108*10,
    ny=6*10,
    nghost=1,
    use_JIT=True,
)
