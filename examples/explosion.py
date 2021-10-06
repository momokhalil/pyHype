from pyHype.solvers import Euler2D

block1 = {'nBLK': 1,
          'NW': [0, 20], 'NE': [10, 20],
          'SW': [0, 0], 'SE': [10, 0],
          'NeighborE': None,
          'NeighborW': None,
          'NeighborN': None,
          'NeighborS': None,
          'BCTypeE': 'Reflection',
          'BCTypeW': 'Reflection',
          'BCTypeN': 'Reflection',
          'BCTypeS': 'Reflection'}

mesh = {1: block1}

# Solver settings
settings = {'problem_type':             'explosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'nozzle',
            'write_every_n_timesteps':  40,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  0.05,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       50,
            'ny':                       100,
            'nghost':                   1,
            }

# Create solver
exp = Euler2D(fvm='FirstOrder',
              gradient='GreenGauss',
              flux_function='Roe',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings,
              mesh=mesh)

# Solve
exp.solve()
