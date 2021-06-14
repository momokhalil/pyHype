from pyHype.solvers import solver

explosion = {'problem_type':            'explosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Conservative',
             'interface_interpolation': 'arithmetic_average',
             'finite_volume_method':    'SecondOrderPWL',
             'gradient_method':         'GreenGauss',
             'flux_limiter':            'Venkatakrishnan',
             'time_integrator':         'RK2',
             'CFL':                     0.4,
             't_final':                 0.002,
             'realplot':                False,
             'makeplot':                False,
             'time_it':                 False,
             'gamma':                   1.4,
             'rho_inf':                 1.0,
             'a_inf':                   343.0,
             'R':                       287.0,
             'nx':                      200,
             'ny':                      400,
             'nghost':                  1,
             'mesh_name':               'chamber',
             'profile':                 True}

exp = solver.Euler2D(explosion)
exp.solve()

exp.profile_data.sort_stats('tottime').print_stats()
