from pyHype.solvers import Euler2D

block1 = {
    "nBLK": 1,
    "NW": [0, 20],
    "NE": [10, 20],
    "SW": [0, 0],
    "SE": [10, 0],
    "NeighborE": None,
    "NeighborW": None,
    "NeighborN": None,
    "NeighborS": None,
    "BCTypeE": "Reflection",
    "BCTypeW": "Reflection",
    "BCTypeN": "Reflection",
    "BCTypeS": "Reflection",
}

mesh = {1: block1}


# Solver settings
settings = {
    "problem_type": "explosion",
    "interface_interpolation": "arithmetic_average",
    "reconstruction_type": "conservative",
    "upwind_mode": "primitive",
    "write_solution": False,
    "write_solution_mode": "every_n_timesteps",
    "write_solution_name": "nozzle",
    "write_every_n_timesteps": 40,
    "plot_every": 20,
    "CFL": 0.8,
    "t_final": 0.07,
    "realplot": False,
    "profile": True,
    "gamma": 1.4,
    "rho_inf": 1.0,
    "a_inf": 343.0,
    "R": 287.0,
    "nx": 600,
    "ny": 600,
    "nghost": 1,
    "use_JIT": True,
}

# Create solver
exp = Euler2D(
    fvm_type="MUSCL",
    fvm_spatial_order=2,
    fvm_num_quadrature_points=1,
    fvm_gradient_type="GreenGauss",
    fvm_flux_function="Roe",
    fvm_slope_limiter="Venkatakrishnan",
    time_integrator="RK4",
    settings=settings,
    mesh_inputs=mesh,
)

# Solve
exp.solve()
