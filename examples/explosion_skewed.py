from pyHype.solvers import Euler2D

# Solver settings
settings = {'problem_type':             'explosion_3',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'primitive',
            'upwind_mode':              'primitive',
            'write_solution':           True,
            'write_solution_mode':      'every_n_timesteps',
            'write_solution_name':      'explosion3',
            'write_every_n_timesteps':  15,
            'CFL':                      0.6,
            't_final':                  0.04,
            'realplot':                 False,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       350,
            'ny':                       350,
            'nghost':                   1,
            'mesh_name':                'chamber_skewed_2'
            }

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='HLLL',
              limiter='Venkatakrishnan',
              integrator='RK3SSP',
              settings=settings)

# Solve
exp.solve()
