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

import sys
import pstats
import cProfile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pyHype import execution_prints
from pyHype.blocks.base import Blocks
from pyHype.mesh import meshes
from pyHype.mesh.base import BlockDescription

np.set_printoptions(threshold=sys.maxsize)


__REQUIRED__ = ['problem_type', 'IC_type', 'realplot', 'makeplot', 'time_it', 't_final', 'time_integrator',
                'flux_function', 'CFL', 'flux_function', 'reconstruction_type', 'finite_volume_method', 'flux_limiter',
                'gamma', 'R', 'rho_inf', 'a_inf', 'nx', 'ny', 'nghost', 'mesh_name', 'profile',
                'interface_interpolation']

__OPTIONAL__ = ['alpha', 'write_time']


class ProblemInput:
    def __init__(self,
                 input_dict: dict,
                 mesh_dict: dict
                 ) -> None:
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
    def _check_input_dict(input_dict: dict) -> None:
        for key in __REQUIRED__:
            if key not in input_dict.keys():
                raise KeyError(key + ' not found in inputs.')


class Euler2DSolver:
    def __init__(self, input_dict: dict) -> None:

        # --------------------------------------------------------------------------------------------------------------
        # Store mesh features required to create block descriptions

        # Mesh name
        mesh_name = input_dict['mesh_name']
        # Number of nodes in x-direction per block
        nx = input_dict['nx']
        # Number of nodes in y-direction per block
        ny = input_dict['ny']

        # --------------------------------------------------------------------------------------------------------------
        # Create dictionary that describes each block in mesh

        # Get function that creates the dictionary of block description dictionaries.
        _mesh_func = meshes.DEFINED_MESHES[mesh_name]
        # Call mesh_func with nx, and ny to return the dictionary of description dictionaries
        _mesh_dict = _mesh_func(nx=nx, ny=ny)
        # Initialise dictionary to store a BlockDescription for each block in the mesh
        _mesh_desc = {}
        # Create BlockDescription for each block in the mesh
        for blk, blkData in _mesh_dict.items():
            _mesh_desc[blk] = BlockDescription(blkData)
        # Create ProblemInput to store inputs and mesh description
        self.inputs = ProblemInput(input_dict, _mesh_desc)

        print(execution_prints.pyhype)
        print(execution_prints.lice)
        print(execution_prints.began_solving + self.inputs.problem_type)
        print('Date and time: ', datetime.today())

        # Create Blocks
        self._blocks = Blocks(self.inputs)

        # --------------------------------------------------------------------------------------------------------------
        # Initialise attributes

        # Simulation time
        self.t = 0
        # Number of time steps
        self.numTimeStep = 0
        # CFL number
        self.CFL = self.inputs.CFL
        # Normalized target simulation time
        self.t_final = self.inputs.t_final * self.inputs.a_inf
        # Profiler results
        self.profile_data = None
        # Real-time plot
        self.realplot = None
        # Plot
        self.plot = None

    @property
    def blocks(self):
        return self._blocks.blocks.values()

    def set_IC(self):

        problem_type = self.inputs.problem_type
        g = self.inputs.gamma
        ny = self.inputs.ny
        nx = self.inputs.nx

        print('    Initial condition type: ', problem_type)

        if problem_type == 'shockbox':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400.0
            uL = 0.0
            vL = 0.0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 0.0
            vR = 0.0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self.blocks:
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
                        elif block.mesh.x[i, j] > 5 and block.mesh.y[i, j] > 5:
                            block.state.U[i, j, :] = QR
                        else:
                            block.state.U[i, j, :] = QL
                block.state.non_dim()

        elif problem_type == 'implosion':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400.0
            uL = 0.00
            vL = 0.0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 0.00
            vR = 0.0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self.blocks:
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 3 and block.mesh.y[i, j] <= 3:
                            block.state.U[i, j, :] = QR
                        else:
                            block.state.U[i, j, :] = QL
                block.state.non_dim()

        elif problem_type == 'explosion':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400.0
            uL = 0.0
            vL = 0.0
            eL = pL / (g - 1) + rhoL * (uL**2 + vL**2) / 2

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 0.00
            vR = 0.0
            eR = pR / (g - 1) + rhoR * (uR**2 + vR**2) / 2

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

            # Fill state vector in each block
            for block in self.blocks:
                for i in range(ny):
                    for j in range(nx):
                        if 3 <= block.mesh.x[i, j] <= 7 and 3 <= block.mesh.y[i, j] <= 7:
                            block.state.U[i, j, :] = QL
                        else:
                            block.state.U[i, j, :] = QR
                block.state.non_dim()

    def set_BC(self):
        self._blocks.set_BC()

    def dt(self):
        dt = 1000000
        for block in self.blocks:
            a = block.state.a()

            t1 = block.mesh.dx / (np.absolute(block.state.u()) + a)
            t2 = block.mesh.dy / (np.absolute(block.state.v()) + a)

            dt_ = self.CFL * min(t1.min(), t2.min())

            if dt_ < dt: dt = dt_

        return dt

    def solve(self):

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Initial Conditions')
        self.set_IC()

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Boundary Conditions')
        self.set_BC()

        if self.inputs.realplot:
            plt.ion()
            self.realplot = plt.axes()
            self.realplot.figure.set_size_inches(5, 10)

        if self.inputs.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        print('Start simulation')
        while self.t < self.t_final:

            dt = self.dt()
            self.numTimeStep += 1

            # print('update block')
            self._blocks.update(dt)

            #self.write_output_nodes('./test_sim/test_sim_U_' + str(self.numTimeStep), self._blocks.blocks[1].state.U)

            if self.inputs.realplot:
                if self.numTimeStep % 1 == 0:
                    self.realplot.contourf(self._blocks.blocks[1].mesh.x,
                                           self._blocks.blocks[1].mesh.y,
                                           self._blocks.blocks[1].state.rho,
                                           100, cmap='magma')
                    plt.show()
                    plt.pause(0.001)

            self.t += dt

        if self.inputs.makeplot:
            self.plot = plt.axes()
            self.plot.figure.set_size_inches(8, 8)
            self.plot.contourf(self._blocks.blocks[1].mesh.x,
                               self._blocks.blocks[1].mesh.y,
                               self._blocks.blocks[1].state.U[:, :, 0],
                               100, cmap='magma')
            plt.show(block=True)

        if self.inputs.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)

    @staticmethod
    def write_output_nodes(filename: str, array: np.ndarray):
        np.save(file=filename, arr=array)
