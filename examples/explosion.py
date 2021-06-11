from pyHype.solvers import solver
import numpy as np
import matplotlib.pyplot as plt

explosion = {'problem_type':            'explosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Conservative',
             'interface_interpolation': 'arithmetic_average',
             'realplot':                True,
             'makeplot':                False,
             'time_it':                 False,
             't_final':                 0.06,
             'CFL':                     0.4,
             'time_integrator':         'RK2',
             'finite_volume_method':    'SecondOrderGreenGauss_EXP',
             'flux_limiter':            'Venkatakrishnan',
             'gamma':                   1.4,
             'rho_inf':                 1.0,
             'a_inf':                   343.0,
             'R':                       287.0,
             'nx':                      101,
             'ny':                      201,
             'nghost':                  1,
             'mesh_name':               'chamber',
             'profile':                 False}

exp = solver.Euler2DSolver(explosion)
exp.solve()

