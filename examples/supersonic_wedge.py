from pyHype.solvers import Euler2D
import numpy as np

block1 = {'nBLK': 1,
          'NW': [0, 2], 'NE': [2, 2],
          'SW': [0, 0], 'SE': [2, 0],
          'NeighborE': 2,
          'NeighborW': None,
          'NeighborN': None,
          'NeighborS': None,
          'BCTypeE': 'None',
          'BCTypeW': 'InletDirichlet',
          'BCTypeN': 'OutletDirichlet',
          'BCTypeS': 'Slipwall'}

block2 = {'nBLK': 2,
          'NW': [2, 2], 'NE': [4, 2],
          'SW': [2, 0], 'SE': [4, 2 * np.tan(15 * np.pi / 180)],
          'NeighborE': None,
          'NeighborW': 1,
          'NeighborN': None,
          'NeighborS': None,
          'BCTypeE': 'OutletDirichlet',
          'BCTypeW': 'None',
          'BCTypeN': 'OutletDirichlet',
          'BCTypeS': 'Slipwall'}

mesh = {1: block1,
        2: block2,
        }

# Solver settings
settings = {'problem_type':             'supersonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'conservative',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'super_wedge',
            'write_every_n_timesteps':  20,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  25.0,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       50,
            'ny':                       50,
            'nghost':                   1,
            'BC_inlet_west_rho':        1.0,
            'BC_inlet_west_u':          2.0,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          1 / 1.4,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='Roe',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings,
              mesh=mesh)

# Solve
exp.solve()
