from pyHype.solvers import Euler2D
from pyHype.mesh.base import QuadMeshGenerator

BCE = ['OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet', 'OutletDirichlet']
BCW = ['Slipwall', 'Slipwall', 'InletDirichlet', 'Slipwall', 'Slipwall']
BCN = ['OutletDirichlet']
BCS = ['OutletDirichlet']

_mesh = QuadMeshGenerator(nx_blk=1, ny_blk=5,
                          BCE=BCE, BCW=BCW, BCN=BCN, BCS=BCS,
                          NE=(1, 0.5), SW=(0, 0), NW=(0, 0.5), SE=(1, 0))

# Solver settings
settings = {'problem_type':             'subsonic_rest',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'conservative',
            'write_solution':           False,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'kvi',
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
            'ny':                       10,
            'nghost':                   1,
            'BC_inlet_west_rho':        1.0,
            'BC_inlet_west_u':          0.25,
            'BC_inlet_west_v':          0.0,
            'BC_inlet_west_p':          2.0 / 1.4,
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings,
              mesh=_mesh)

# Solve
exp.solve()