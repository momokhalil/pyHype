import numpy as np
from pyhype.solvers import Euler2D
from pyhype.solvers.base import ProblemInput
from pyhype.mesh.base import QuadMeshGenerator
from pyhype.states import PrimitiveState, State


def inlet_diriclet_bc(state: State):
    BC_inlet_west_rho = 1.0
    BC_inlet_west_u = 0.25
    BC_inlet_west_v = 0.0
    BC_inlet_west_p = 2.0 / 1.4

    inlet_state = PrimitiveState(
        inputs=state.inputs,
        array=np.array(
            [
                BC_inlet_west_rho,
                BC_inlet_west_u,
                BC_inlet_west_v,
                BC_inlet_west_p,
            ]
        ).reshape((1, 1, 4)),
    )
    state.from_state(inlet_state)


BCE = [
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
    "OutletDirichlet",
]
BCW = ["Slipwall", "Slipwall", inlet_diriclet_bc, "Slipwall", "Slipwall"]
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
inputs = ProblemInput(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="HLLL",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK2",
    mesh=_mesh,
    problem_type="subsonic_rest",
    interface_interpolation="arithmetic_average",
    reconstruction_type=PrimitiveState,
    write_solution=False,
    write_solution_mode="every_n_timesteps",
    write_solution_name="kvi",
    write_every_n_timesteps=20,
    plot_every=10,
    CFL=0.4,
    t_final=25.0,
    realplot=True,
    profile=False,
    gamma=1.4,
    rho_inf=1.0,
    a_inf=1.0,
    R=287.0,
    nx=100,
    ny=10,
    nghost=1,
    use_JIT=True,
)

# Create solver
exp = Euler2D(inputs=inputs)

# Solve
exp.solve()
