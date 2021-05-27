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
                'gamma', 'R', 'rho_inf', 'a_inf', 'nx', 'ny', 'mesh_name', 'profile']

__OPTIONAL__ = ['alpha', 'write_time']


class ProblemInput:
    def __init__(self, input_dict: dict, mesh_dict: dict):
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # Check input dictionary to check if all required fields are present
        self._check_input_dict(input_dict)

        # REQUIRED

        # General parameters
        for req_name in __REQUIRED__:
            self.__setattr__(req_name, input_dict[req_name])

        self.n = input_dict['nx'] * input_dict['ny']
        self.mesh_inputs = mesh_dict

        # OPTIONAL
        for opt_name in __OPTIONAL__:
            if opt_name in input_dict.keys():
                self.__setattr__(opt_name, input_dict[opt_name])

    @staticmethod
    def _check_input_dict(input_dict):
        for key in __REQUIRED__:
            if key not in input_dict.keys():
                raise KeyError(key + ' not found in inputs.')
