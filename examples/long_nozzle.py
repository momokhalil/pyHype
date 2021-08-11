from pyHype.solvers.solver import Euler2D

# Solver settings
settings = {'problem_type':             'subsonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'CFL':                      0.4,
            't_final':                  0.05,
            'realplot':                 False,
            'profile':                  True,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       70,
            'ny':                       70,
            'nghost':                   1,
            'mesh_name':                'long_nozzle'}

# Create solver
exp = Euler2D(fvm='SecondOrderPWL',
              gradient='GreenGauss',
              flux_function='Roe',
              limiter='Venkatakrishnan',
              integrator='RK2',
              settings=settings)

# Solve
exp.solve()
