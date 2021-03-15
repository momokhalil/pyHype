import mesh_inputs

def E4():
    _E4 = {'problem_type': 'E4',
           'IC_type': 'from_IC',
           'flux_function': 'Roe',
           'realplot': 1,
           'nondim': 1,
           'make_plot': 1,
           'time_it': 1,
           't_final': 7/1000,
           'eps': 1e-8,
           'time_integrator': 'RK4',
           'CFL': 0.95,
           'reconstruction_order': 2,
           'flux_limiter': 'vanAlbada',
           'gamma': 1.4,
           'rho_inf': 1,
           'a_inf': 343,
           'R': 287,
           'mesh_inputs': mesh_inputs.simple_mesh()}

    return _E4


