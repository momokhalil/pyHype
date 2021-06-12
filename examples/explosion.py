from pyHype.solvers import solver
import numpy as np
import matplotlib.pyplot as plt

explosion = {'problem_type':            'explosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Primitive',
             'interface_interpolation': 'arithmetic_average',
             'finite_volume_method':    'SecondOrderPWL',
             'gradient_method':         'GreenGauss',
             'flux_limiter':            'Venkatakrishnan',
             'time_integrator':         'RK2',
             'CFL':                     0.4,
             't_final':                 0.06,
             'realplot':                False,
             'makeplot':                False,
             'time_it':                 False,
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

