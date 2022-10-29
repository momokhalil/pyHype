from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import ConservativeState
from pyhype.solvers.base import ProblemInput
from pyhype.initial_conditions.shockbox import ShockboxInitialCondition

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

block1 = {
    "nBLK": 1,
    "NW": [0, 10],
    "NE": [10, 10],
    "SW": [0, 0],
    "SE": [10, 0],
    "NeighborE": None,
    "NeighborW": None,
    "NeighborN": None,
    "NeighborS": None,
    "NeighborNE": None,
    "NeighborNW": None,
    "NeighborSE": None,
    "NeighborSW": None,
    "BCTypeE": "Reflection",
    "BCTypeW": "Reflection",
    "BCTypeN": "Reflection",
    "BCTypeS": "Reflection",
    "BCTypeNE": None,
    "BCTypeNW": None,
    "BCTypeSE": None,
    "BCTypeSW": None,
}

mesh = {1: block1}

# Solver settings
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="Roe",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK2",
    mesh=mesh,
    initial_condition=ShockboxInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=ConservativeState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="shockbox",
    write_every_n_timesteps=30,
    plot_every=10,
    CFL=0.4,
    t_final=2.0,
    realplot=True,
    profile=False,
    fluid=air,
    nx=50,
    ny=50,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(inputs=inputs)

exp.solve()
