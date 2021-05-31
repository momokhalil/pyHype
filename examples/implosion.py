from pyHype.solvers import solver
import os

implosion = {'problem_type':            'implosion',
             'IC_type':                 'from_IC',
             'flux_function':           'Roe',
             'reconstruction_type':     'Conservative',
             'realplot':                1,
             'makeplot':                1,
             'time_it':                 1,
             't_final':                 0.02,
             'time_integrator':         'RK2',
             'CFL':                     0.5,
             'finite_volume_method':    'SecondOrderGreenGauss',
             'flux_limiter':            'van_albada',
             'gamma':                   1.4,
             'rho_inf':                 1.0,
             'a_inf':                   343.0,
             'R':                       287.0,
             'nx':                      100,
             'ny':                      100,
             'nghost':                  1,
             'mesh_name':               'one_mesh'}

os.environ["NUMBA_DISABLE_JIT"] = str(0)

imp = solver.Euler2DSolver(implosion)
imp.solve()
