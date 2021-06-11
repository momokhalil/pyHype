from pyHype.solvers import solver
import os

implosion = {'problem_type':            'implosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Conservative',
             'interface_interpolation': 'arithmetic_average',
             'realplot':                True,
             'makeplot':                False,
             'time_it':                 False,
             't_final':                 0.02,
             'time_integrator':         'RK3SSP',
             'CFL':                     0.4,
             'finite_volume_method':    'SecondOrderGreenGauss',
             'flux_limiter':            'VanAlbada',
             'gamma':                   1.4,
             'rho_inf':                 1.0,
             'a_inf':                   343.0,
             'R':                       287.0,
             'nx':                      100,
             'ny':                      100,
             'nghost':                  1,
             'mesh_name':               'one_mesh',
             'profile':                 False}

imp = solver.Euler2DSolver(implosion)
imp.solve()
