from pyhype.fluids import Air
from pyhype.solver_config import SolverConfig
from pyhype.states import PrimitiveState
from pyhype.initial_conditions.supersonic_flood import SupersonicFloodInitialCondition

air = Air(a_inf=343.0, rho_inf=1.0)

initial_condition = SupersonicFloodInitialCondition(
    fluid=air,
    rho=1.0,
    u=5.0,
    v=0.0,
    p=1 / air.gamma(),
)

# Solver settings
config = SolverConfig(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="HLLL",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    initial_condition=initial_condition,
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=True,
    write_solution_mode="every_n_timesteps",
    write_solution_name="super_step",
    write_solution_base="C:\\Users\Mohamed\Desktop\cfd\pyHype-main\examples\supersonic_step\data",
    write_every_n_timesteps=50,
    plot_every=10,
    CFL=0.3,
    t_final=20.0,
    realplot=False,
    profile=True,
    fluid=air,
    nx=48 * 4,
    ny=16 * 4,
    nghost=1,
    use_JIT=True,
)
