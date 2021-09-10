from pyHype.solvers import Euler2D
from pyHype.mesh.base import QuadMeshGenerator

_left_x = [0.0, 0.0]
_left_y = [0.0, 1.0]
_right_x = [4.5, 4.5]
_right_y = [0.34, 0.6]
_top_x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
_bot_x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
_top_y = [1.0, 1.0, 0.94, 0.85, 0.70, 0.63, 0.60, 0.6, 0.6, 0.6]
_bot_y = [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.25, 0.32, 0.34, 0.34]

BCS = ['OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet',
       'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall']

BCN = ['OutletDirichlet', 'Slipwall', 'Slipwall', 'Slipwall',
       'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall', 'Slipwall']

_mesh = QuadMeshGenerator(nx_blk=9, ny_blk=1,
                          BCE='OutletDirichlet', BCW='InletDirichlet', BCN=BCN, BCS=BCS,
                          top_x=_top_x, bot_x=_bot_x, top_y=_top_y, bot_y=_bot_y,
                          left_x=_left_x, right_x=_right_x, left_y=_left_y, right_y=_right_y)

# Solver settings
settings = {'problem_type':             'supersonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'ramjet',
            'write_every_n_timesteps':  40,
            'plot_every':               10,
            'CFL':                      0.4,
            't_final':                  10.0,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       20,
            'ny':                       20,
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
              mesh=_mesh)
# Solve
exp.solve()
