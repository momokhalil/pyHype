import numpy as np
from pyhype.fluids import Air
from pyhype.solvers import Euler2D
from pyhype.states import PrimitiveState
from pyhype.solvers.base import SolverConfig
from pyhype.mesh.base import QuadMeshGenerator
from pyhype.boundary_conditions.base import PrimitiveDirichletBC
from examples.jet.initial_condition import SubsonicJetInitialCondition

# Define fluid
air = Air(a_inf=343.0, rho_inf=1.0)

inlet_rho = 1.0
inlet_u = 0.1
inlet_v = 0.0
inlet_p = 2.0 / air.gamma()

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
subsonic_inlet_bc = PrimitiveDirichletBC(primitive_state=inlet_state)

BCE = [
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
]
BCW = ["Slipwall", "Slipwall", subsonic_inlet_bc, "Slipwall", "Slipwall"]
BCN = ["OutletDirichlet"]
BCS = ["OutletDirichlet"]

_mesh = QuadMeshGenerator(
    nx_blk=1,
    ny_blk=5,
    BCE=BCE,
    BCW=BCW,
    BCN=BCN,
    BCS=BCS,
    NE=(1, 0.5),
    SW=(0, 0),
    NW=(0, 0.5),
    SE=(1, 0),
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
    mesh=_mesh,
    initial_condition=SubsonicJetInitialCondition(),
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="kvi",
    write_every_n_timesteps=20,
    plot_every=30,
    CFL=0.4,
    t_final=25.0,
    realplot=True,
    profile=False,
    fluid=air,
    nx=100,
    ny=10,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(config=config)

# Solve
exp.solve()
