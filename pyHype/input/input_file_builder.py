"""
Copyright 2021 Mohamed Khalil

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

__REQUIRED__ = ['problem_type', 'IC_type', 'realplot', 'makeplot', 'time_it', 't_final', 'time_integrator',
                'flux_function', 'CFL', 'flux_function', 'reconstruction_type', 'finite_volume_method', 'flux_limiter',
                'gamma', 'R', 'rho_inf', 'a_inf', 'nx', 'ny', 'mesh_name']

__OPTIONAL__ = ['alpha']


class ProblemInput:
    def __init__(self, input_dict: dict, mesh_dict: dict):
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # Check input dictionary to check if all required fields are present
        self._check_input_dict(input_dict)

        # REQUIRED

        # General parameters
        self.problem_type = input_dict['problem_type']
        self.IC_type = input_dict['IC_type']
        self.realplot = input_dict['realplot']
        self.makeplot = input_dict['makeplot']
        self.time_it = input_dict['time_it']
        self.t_final = input_dict['t_final']

        # Numerical method parameters
        self.time_integrator = input_dict['time_integrator']
        self.CFL = input_dict['CFL']
        self.flux_function = input_dict['flux_function']
        self.reconstruction_type = input_dict['reconstruction_type']
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


        # OPTIONAL
        if 'alpha' in input_dict.keys():
            self.alpha = input_dict['alpha']

    @staticmethod
    def _check_input_dict(input_dict):
        for key in __REQUIRED__:
            if key not in input_dict.keys():
                raise KeyError(key + ' not found in inputs.')
