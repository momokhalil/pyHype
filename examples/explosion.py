from pyHype.solvers import solver

explosion = {'problem_type': 'explosion',
             'IC_type': 'from_IC',
             'flux_function': 'Roe',
             'reconstruction_type': 'Primitive',
             'realplot': 0,
             'makeplot': 0,
             'time_it': 0,
             't_final': 0.06,
             'time_integrator': 'RK4',
             'CFL': 0.8,
             'finite_volume_method': 'SecondOrderGreenGauss',
             'flux_limiter': 'van_albada',
             'gamma': 1.4,
             'rho_inf': 1.0,
             'a_inf': 343.0,
             'R': 287.0,
             'nx': 600,
             'ny': 1200,
             'mesh_name': 'chamber',
             'profile': False}

exp = solver.Euler2DSolver(explosion)
exp.solve()
