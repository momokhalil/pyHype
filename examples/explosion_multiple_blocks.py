from pyHype.solvers import solver

# Solver settings
settings = {'problem_type':             'explosion',
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
            'nx':                       80,
            'ny':                       80,
            'nghost':                   1,
            'mesh_name':                'square_ten_by_ten_four_block'}

# Create solver
exp = solver.Euler2D(fvm='SecondOrderPWL',
                     gradient='GreenGauss',
                     flux_function='Roe',
                     limiter='Venkatakrishnan',
                     integrator='RK2',
                     settings=settings)

# Solve
exp.solve()
