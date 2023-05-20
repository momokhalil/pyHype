from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import ConservativeState
from pyhype.solver_config import SolverConfig
from examples.implosion.initial_condition import ImplosionInitialCondition

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

block0 = {
    "nBLK": 0,
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

mesh = {0: block0}

# Solver settings
config = SolverConfig(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function_type="Roe",
    fvm_slope_limiter_type="Venkatakrishnan",
    time_integrator="RK2",
    initial_condition=ImplosionInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=ConservativeState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="nozzle",
    write_every_n_timesteps=15,
    plot_every=10,
    CFL=0.4,
    t_final=0.1,
    realplot=False,
    profile=True,
    fluid=air,
    nx=40,
    ny=40,
    nghost=1,
    use_JIT=True,
)

if __name__ == "__main__":
    implosion = Euler2D(
        config=config,
        mesh_config=mesh,
    )
    implosion.solve()
