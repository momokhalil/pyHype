from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'supersonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'write_solution':           True,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'ramjet',
            'write_every_n_timesteps':  40,
            'CFL':                      0.4,
            't_final':                  10.0,
            'realplot':                 False,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       300,
            'ny':                       300,
            'nghost':                   1,
            'mesh_name':                'ramjet',
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
              settings=settings)
# Solve
exp.solve()
