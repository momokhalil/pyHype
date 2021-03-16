import mesh_inputs

def E4():
    nx = 4
    ny = 4
    n = nx * ny

    _E4 = {'problem_type': 'shockbox',
           'IC_type': 'from_IC',
           'flux_function': 'Roe',
           'realplot': 1,
           'nondim': 1,
           'make_plot': 1,
           'time_it': 1,
           't_final': 7 / 1000,
           'eps': 1e-8,
           'time_integrator': 'RK4',
           'CFL': 0.95,
           'finite_volume_method': 'SecondOrderLimited',
           'flux_limiter': 'vanAlbada',
           'gamma': 1.4,
           'rho_inf': 1,
           'a_inf': 343,
           'R': 287,
           'nx': nx,
           'ny': ny,
           'n': n,
           'mesh_inputs': mesh_inputs.simple_mesh(nx, ny, n)}

    return _E4
