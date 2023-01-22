import numpy as np
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import PrimitiveState
from pyhype.solvers.base import SolverConfig
from pyhype.boundary_conditions.base import PrimitiveDirichletBC
from pyhype.initial_conditions.supersonic_flood import SupersonicFloodInitialCondition

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

initial_condition = SupersonicFloodInitialCondition(
    fluid=air,
    rho=1.0,
    u=2.0,
    v=0.0,
    p=1 / air.gamma(),
)

# Define inlet BC
inlet_rho = 1.0
inlet_u = 2.0
inlet_v = 0.0
inlet_p = 1 / air.gamma()

inlet_state = PrimitiveState(
    fluid=air,
    array=np.array(
        [
            inlet_rho,
            inlet_u,
            inlet_v,
            inlet_p,
        ]
    ).reshape((1, 1, 4)),
)

mach_2_wedge_inlet_bc = PrimitiveDirichletBC(primitive_state=inlet_state)

# Define mesh
block1 = {
    "nBLK": 1,
    "NW": [0, 2],
    "NE": [2, 2],
    "SW": [0, 0],
    "SE": [2, 0],
    "NeighborE": 2,
    "NeighborW": None,
    "NeighborN": None,
    "NeighborS": None,
    "NeighborNE": None,
    "NeighborNW": None,
    "NeighborSE": None,
    "NeighborSW": None,
    "BCTypeE": None,
    "BCTypeW": mach_2_wedge_inlet_bc,
    "BCTypeN": "OutletDirichlet",
    "BCTypeS": "Reflection",
    "BCTypeNE": None,
    "BCTypeNW": None,
    "BCTypeSE": None,
    "BCTypeSW": None,
}

block2 = {
    "nBLK": 2,
    "NW": [2, 2],
    "NE": [4, 2 + 2 * np.tan(15 * np.pi / 180)],
    "SW": [2, 0],
    "SE": [4, 2 * np.tan(15 * np.pi / 180)],
    "NeighborE": None,
    "NeighborW": 1,
    "NeighborN": None,
    "NeighborS": None,
    "NeighborNE": None,
    "NeighborNW": None,
    "NeighborSE": None,
    "NeighborSW": None,
    "BCTypeE": "OutletDirichlet",
    "BCTypeW": None,
    "BCTypeN": "OutletDirichlet",
    "BCTypeS": "Reflection",
    "BCTypeNE": None,
    "BCTypeNW": None,
    "BCTypeSE": None,
    "BCTypeSW": None,
}
mesh = {
    1: block1,
    2: block2,
}

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
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="machref",
    write_every_n_timesteps=20,
    plot_every=20,
    CFL=0.3,
    t_final=20,
    realplot=True,
    profile=False,
    fluid=air,
    nx=60,
    ny=60,
    nghost=1,
    use_JIT=True,
    mesh=mesh,
)

# Create solver
exp = Euler2D(config=config)

# Solve
exp.solve()
