from pyhype.fluids import Air
from pyhype.solver_config import SolverConfig
from pyhype.states import PrimitiveState
from examples.dmr.initial_condition import DMRInitialCondition

air = Air(a_inf=343.0, rho_inf=1.0)

config = SolverConfig(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="HLLL",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    initial_condition=DMRInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="machref",
    write_every_n_timesteps=20,
    plot_every=20,
    CFL=0.4,
    t_final=0.25,
    realplot=True,
    profile=False,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
)
