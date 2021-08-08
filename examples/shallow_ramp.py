from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'supersonic_flood',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'CFL':                      0.4,
            't_final':                  5.0,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    1.0,
            'R':                        287.0,
            'nx':                       50,
            'ny':                       50,
            'nghost':                   1,
            'mesh_name':                'shallow_ramp_two_block'}

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL',
                     gradient='GreenGauss',
                     flux_function='Roe',
                     limiter='Venkatakrishnan',
                     integrator='RK2',
                     settings=settings)

# Solve
exp.solve()
