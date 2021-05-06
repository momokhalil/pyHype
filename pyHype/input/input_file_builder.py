
class ProblemInput:
    def __init__(self, input_dict, mesh_dict):
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # General parameters
        self.problem_type = input_dict['problem_type']
        self.IC_type = input_dict['IC_type']
        self.realplot = input_dict['realplot']
        self.time_it = input_dict['time_it']
        self.t_final = input_dict['t_final']

        # Numerical method parameters
        self.time_integrator = input_dict['time_integrator']
        self.CFL = input_dict['CFL']
        self.flux_function = input_dict['flux_function']
        self.finite_volume_method = input_dict['finite_volume_method']
        self.flux_limiter = input_dict['flux_limiter']

        # Thermodynamic parameters
        self.gamma = input_dict['gamma']
        self.R = input_dict['R']
        self.rho_inf = input_dict['rho_inf']
        self.a_inf = input_dict['a_inf']

        # Mesh parameters
        self.n = input_dict['nx'] * input_dict['ny']
        self.nx = input_dict['nx']
        self.ny = input_dict['ny']
        self.mesh_name = input_dict['mesh_name']
        self.mesh_inputs = mesh_dict
