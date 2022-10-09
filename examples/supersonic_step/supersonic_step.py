from pyHype.solvers import Euler2D
from mesh import step_ten_block

mesh = step_ten_block()

# Solver settings
settings = {
    "problem_type": "supersonic_flood",
    "interface_interpolation": "arithmetic_average",
    "reconstruction_type": "primitive",
    "write_solution": False,
    "write_solution_mode": "every_n_timesteps",
    "write_solution_name": "super_step",
    "write_every_n_timesteps": 20,
    "plot_every": 10,
    "CFL": 0.3,
    "t_final": 25.0,
    "realplot": True,
    "profile": False,
    "gamma": 1.4,
    "rho_inf": 1.0,
    "a_inf": 1.0,
    "R": 287.0,
    "nx": 48,
    "ny": 16,
    "nghost": 1,
    "use_JIT": True,
    "BC_inlet_west_rho": 1.0,
    "BC_inlet_west_u": 5.0,
    "BC_inlet_west_v": 0.0,
    "BC_inlet_west_p": 1 / 1.4,
}


# Create solver
exp = Euler2D(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="HLLL",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK2",
    settings=settings,
    mesh_inputs=mesh,
)

# Solve
exp.solve()
