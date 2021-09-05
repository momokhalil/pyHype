import numpy as np
from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'mach_reflection',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'conservative',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'machref',
            'write_every_n_timesteps':  20,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  0.25,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       50,
            'ny':                       50,
            'nghost':                   1,
            'mesh_name':                'wedge_35_four_block',
            'BC_inlet_west_rho':        8.0,
            'BC_inlet_west_u':          8.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          116.5,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings)

# Solve
exp.solve()
