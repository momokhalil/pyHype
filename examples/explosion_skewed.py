from pyHype.solvers import solver

explo_skw = {'problem_type':            'explosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Conservative',
             'interface_interpolation': 'arithmetic_average',
             'finite_volume_method':    'SecondOrderPWL',
             'gradient_method':         'GreenGauss',
             'flux_limiter':            'Venkatakrishnan',
             'time_integrator':         'RK2',
             'CFL':                     0.4,
             't_final':                 0.07,
             'realplot':                False,
             'makeplot':                False,
             'time_it':                 False,
             'gamma':                   1.4,
             'rho_inf':                 1.0,
             'a_inf':                   343.0,
             'R':                       287.0,
             'nx':                      600,
             'ny':                      1200,
             'nghost':                  1,
             'mesh_name':               'chamber_skewed',
             'profile':                 False}

exp = solver.Euler2D(explo_skw)
exp.solve()
