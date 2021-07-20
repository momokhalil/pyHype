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
from __future__ import annotations

import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

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

from typing import TYPE_CHECKING
from typing import Iterable

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock

np.set_printoptions(threshold=sys.maxsize)


__REQUIRED__ = ['problem_type', 'realplot', 't_final', 'CFL',
                'gamma', 'R', 'rho_inf', 'a_inf', 'nx', 'ny', 'nghost', 'mesh_name', 'profile',
                'interface_interpolation', 'reconstruction_type']

__OPTIONAL__ = ['alpha', 'write_time', 'upwind_mode']


class ProblemInput:
    def __init__(self,
                 fvm: str,
                 gradient: str,
                 flux_function: str,
                 limiter: str,
                 integrator: str,
                 settings: dict,
                 mesh_inputs: dict
                 ) -> None:
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # Check input dictionary to check if all required fields are present
        self._check_input_settings(settings)

        # REQUIRED

        self.fvm = fvm
        self.gradient = gradient
        self.flux_function = flux_function
        self.limiter = limiter
        self.integrator = integrator

        # General parameters
        for req_name in __REQUIRED__:
            self.__setattr__(req_name, settings[req_name])

        self.n = settings['nx'] * settings['ny']
        self.mesh_inputs = mesh_inputs

        # OPTIONAL
        for opt_name in __OPTIONAL__:
            if opt_name in settings.keys():
                self.__setattr__(opt_name, settings[opt_name])

    @staticmethod
    def _check_input_settings(input_dict: dict) -> None:
        for key in __REQUIRED__:
            if key not in input_dict.keys():
                raise KeyError(key + ' not found in inputs.')


class Euler2D:
    def __init__(self,
                 fvm:           str = 'SecondOrderPWL',
                 gradient:      str = 'GreenGauss',
                 flux_function: str = 'Roe',
                 limiter:       str = 'Venkatakrishnan',
                 integrator:    str = 'RK2',
                 settings:      dict = None
                 ) -> None:

        # --------------------------------------------------------------------------------------------------------------
        # Store mesh features required to create block descriptions

        # Mesh name
        mesh_name = settings['mesh_name']
        # Number of nodes in x-direction per block
        nx = settings['nx']
        # Number of nodes in y-direction per block
        ny = settings['ny']
        # Number of ghost cells
        nghost = settings['nghost']

        # --------------------------------------------------------------------------------------------------------------
        # Create dictionary that describes each block in mesh

        # Get function that creates the dictionary of block description dictionaries.
        _mesh_func = meshes.DEFINED_MESHES[mesh_name]
        # Call mesh_func with nx, and ny to return the dictionary of description dictionaries
        _mesh_dict = _mesh_func(nx=nx, ny=ny, nghost=nghost)
        # Initialise dictionary to store a BlockDescription for each block in the mesh
        _mesh_inputs = {}
        # Create BlockDescription for each block in the mesh
        for blk, blkData in _mesh_dict.items():
            _mesh_inputs[blk] = BlockDescription(blkData)
        # Create ProblemInput to store inputs and mesh description
        self.inputs = ProblemInput(fvm=fvm, gradient=gradient, flux_function=flux_function, limiter=limiter,
                                   integrator=integrator, settings=settings, mesh_inputs=_mesh_inputs)

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
        # Time step
        self.dt = 0
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
    def blocks(self) -> Iterable[QuadBlock]:
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

        elif problem_type == 'one_step_shock':

            # High pressure zone
            rhoL = 1.1742
            pL = 101100.0
            uL = 150.0
            vL = 0.0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 150.0
            vR = 0.0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

            # Fill state vector in each block
            for block in self.blocks:
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 1:
                            block.state.U[i, j, :] = QL
                        else:
                            block.state.U[i, j, :] = QR
                block.state.set_vars_from_state()
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
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
                        else:
                            block.state.U[i, j, :] = QL
                block.state.set_vars_from_state()
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
                block.state.set_vars_from_state()
                block.state.non_dim()

        elif problem_type == 'explosion_trapezoid':

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
                        if (-0.8 <= block.mesh.x[i, j] <= -0.2 and -0.8 <= block.mesh.y[i, j] <= -0.2) or \
                           (0.2 <= block.mesh.x[i, j] <= 0.8 and 0.2 <= block.mesh.y[i, j] <= 0.8) or \
                           (-0.8 <= block.mesh.x[i, j] <= -0.2 and 0.2 <= block.mesh.y[i, j] <= 0.8) or \
                           (0.2 <= block.mesh.x[i, j] <= 0.8 and -0.8 <= block.mesh.y[i, j] <= -0.2):
                            block.state.U[i, j, :] = QL
                        else:
                            block.state.U[i, j, :] = QR
                block.state.set_vars_from_state()
                block.state.non_dim()

    def set_BC(self):
        self._blocks.set_BC()

    def get_dt(self):
        """
        Return the time step for all blocks handled by this process based on the CFL condition.

        Parameters:
            - None

        Returns:
            - dt (np.float): Float representing the value of the time step
        """

        return min([block.get_dt() for block in self.blocks])

    def increment_time(self):
        self.t += self.dt

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
            w = max(self.inputs.mesh_inputs[1].SE[0] - self.inputs.mesh_inputs[1].SW[0],
                    self.inputs.mesh_inputs[1].NE[0] - self.inputs.mesh_inputs[1].NW[0])
            l = max(self.inputs.mesh_inputs[1].SE[1] - self.inputs.mesh_inputs[1].NE[1],
                    self.inputs.mesh_inputs[1].NW[1] - self.inputs.mesh_inputs[1].SW[1])
            a = l/w
            pl = 10
            self.realplot.figure.set_size_inches(pl/a, pl)

        if self.inputs.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        print('Start simulation')
        while self.t < self.t_final:

            print(self.t)

            self.dt = self.get_dt()
            self.numTimeStep += 1

            # print('update block')
            self._blocks.update(self.dt)

            ############################################################################################################
            # THIS IS FOR DEBUGGING PURPOSES ONLY
            if self.inputs.realplot:
                if self.numTimeStep % 1 == 0:

                    max_ = max([np.max(block.state.rho) for block in self.blocks])
                    min_ = min([np.min(block.state.rho) for block in self.blocks])

                    for block in self.blocks:
                        """U = block.get_nodal_solution(interpolation='cell_average',
                                                     formulation='conservative')"""
                        self.realplot.contourf(block.mesh.x[:, :, 0],
                                               block.mesh.y[:, :, 0],
                                               block.state.rho,
                                               20, cmap='magma', vmax=max_, vmin=min_)
                    plt.show()
                    plt.pause(0.001)
            ############################################################################################################

            # Increment simulation time
            self.increment_time()

        if self.inputs.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats('tottime').print_stats()

        print('Date and time: ', datetime.today())

    @staticmethod
    def write_output_nodes(filename: str, array: np.ndarray):
        np.save(file=filename, arr=array)
