from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'explosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'Conservative',
            'CFL':                      0.4,
            't_final':                  0.02,
            'realplot':                 False,
            'makeplot':                 False,
            'time_it':                  False,
            'profile':                  True,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       150,
            'ny':                       300,
            'nghost':                   1,
            'mesh_name':                'chamber'}

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL', gradient='GreenGauss', flux_function='Roe',
                     limiter='Venkatakrishnan', integrator='RK2', settings=settings)

# Solve
exp.solve()
