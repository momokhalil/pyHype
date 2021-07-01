from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'implosion',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'Conservative',
            'upwind_mode':              'primitive',
            'CFL':                      0.4,
            't_final':                  0.007,
            'realplot':                 False,
            'profile':                  True,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       200,
            'ny':                       200,
            'nghost':                   1,
            'mesh_name':                'one_mesh'
            }

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL',
                     gradient='GreenGauss',
                     flux_function='Roe',
                     limiter='Venkatakrishnan',
                     integrator='RK2',
                     settings=settings)
# Solve
exp.solve()
