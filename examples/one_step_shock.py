from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'one_step_shock',
            'interface_interpolation':  'arithmetic_average',
            'reconstruction_type':      'conservative',
            'upwind_mode':              'primitive',
            'CFL':                      0.4,
            't_final':                  0.05,
            'realplot':                 True,
            'profile':                  False,
            'gamma':                    1.4,
            'rho_inf':                  1.0,
            'a_inf':                    343.0,
            'R':                        287.0,
            'nx':                       100,
            'ny':                       100,
            'nghost':                   1,
            'mesh_name':                'ramp_three_block'}

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL',
                     gradient='GreenGauss',
                     flux_function='Roe',
                     limiter='Venkatakrishnan',
                     integrator='RK2',
                     settings=settings)

# Solve
exp.solve()
