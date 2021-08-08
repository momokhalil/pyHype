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
import pyHype.solvers.initial_conditions.initial_conditions as ic

from typing import TYPE_CHECKING
from typing import Iterable

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock

np.set_printoptions(threshold=sys.maxsize)


__REQUIRED__ = ['problem_type', 'realplot', 't_final', 'CFL',
                'gamma', 'R', 'rho_inf', 'a_inf', 'nx', 'ny', 'nghost', 'mesh_name', 'profile',
                'interface_interpolation', 'reconstruction_type', 'write_solution']

__OPTIONAL__ = ['alpha', 'write_time', 'upwind_mode', 'write_every_n_timesteps', 'write_solution_mode',
                'write_solution_name']

_DEFINED_IC_ = ['explosion',
                'implosion',
                'shockbox',
                'supersonic_flood',
                'supersonic_rest',
                'subsonic_flood',
                'subsonic_rest',
                'explosion_trapezoid'
                ]


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

        # OPTIONAL
        for opt_name in __OPTIONAL__:
            if opt_name in settings.keys():
                self.__setattr__(opt_name, settings[opt_name])

        self.n = settings['nx'] * settings['ny']
        self.mesh_inputs = mesh_inputs

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
        self.realfig, self.realplot = None, None
        # Plot
        self.plot = None

    @property
    def blocks(self) -> Iterable[QuadBlock]:
        return self._blocks.blocks.values()

    def set_IC(self):

        if self.inputs.problem_type not in _DEFINED_IC_:
            raise ValueError('Initial condition of type ' + str(self.inputs.problem_type) + ' has not been specialized.'
                             ' Please make sure it is defined in ./initial_conditions/initial_conditions.py and added'
                             'to the list of defined ICs in _DEFINED_IC_ on top of this file.')
        else:

            problem_type = self.inputs.problem_type

            print('    Initial condition type: ', problem_type)

            if problem_type == 'implosion':
                _set_IC = ic.implosion(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'explosion':
                _set_IC = ic.explosion(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'shockbox':
                _set_IC = ic.shockbox(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'supersonic_flood':
                _set_IC = ic.supersonic_flood(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'supersonic_rest':
                _set_IC = ic.supersonic_rest(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'subsonic_flood':
                _set_IC = ic.subsonic_flood(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'subsonic_rest':
                _set_IC = ic.subsonic_rest(self.blocks, g=self.inputs.gamma)
            elif problem_type == 'explosion_trapezoid':
                _set_IC = ic.explosion_trapezoid(self.blocks, g=self.inputs.gamma)

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

        """fig, ax = plt.subplots(1)
        for block in self.blocks:
            block.plot(ax=ax)
        plt.show(block=True)"""

        if self.inputs.realplot:
            plt.ion()
            self.realfig, self.realplot = plt.subplots(1)

            blks = [blk for blk in self.inputs.mesh_inputs.values()]

            sw_x = min([blk.SW[0] for blk in blks])
            nw_x = min([blk.NW[0] for blk in blks])

            se_x = max([blk.SE[0] for blk in blks])
            ne_x = max([blk.NE[0] for blk in blks])

            sw_y = min([blk.SW[1] for blk in blks])
            se_y = min([blk.SE[1] for blk in blks])

            nw_y = max([blk.NW[1] for blk in blks])
            ne_y = max([blk.NE[1] for blk in blks])

            ymax = max(nw_y, ne_y)
            ymin = min(sw_y, se_y)

            xmax = max(se_x, ne_x)
            xmin = min(sw_x, nw_x)

            W = xmax - xmin
            L = ymax - ymin

            a = L/W
            pl = 5

            self.realplot.figure.set_size_inches(pl/a, pl)

        if self.inputs.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        for block in self.blocks:
            self.write_output_nodes('./mesh_blk_x_' + str(block.global_nBLK), block.mesh.x)
            self.write_output_nodes('./mesh_blk_y_' + str(block.global_nBLK), block.mesh.y)


        print('Start simulation')
        while self.t < self.t_final:

            print(self.t / self.inputs.a_inf)

            self.dt = self.get_dt()

            # print('update block')
            self._blocks.update(self.dt)

            ############################################################################################################
            # THIS IS FOR DEBUGGING PURPOSES ONLY
            if self.inputs.write_solution:
                if self.inputs.write_solution_mode == 'every_n_timesteps':
                    if self.numTimeStep % self.inputs.write_every_n_timesteps == 0:
                        for block in self.blocks:
                            self.write_output_nodes('./' + self.inputs.write_solution_name
                                                         + '_' + str(self.numTimeStep)
                                                         + '_blk_' + str(block.global_nBLK),
                                                    block.state.U)

            if self.inputs.realplot:
                if self.numTimeStep % 10 == 0:
                    _v = [block.state.Ma() for block in self.blocks]
                    max_ = max([np.max(v) for v in _v])
                    min_ = min([np.min(v) for v in _v])

                    for block in self.blocks:
                        self.realplot.contourf(block.mesh.x[:, :, 0],
                                               block.mesh.y[:, :, 0],
                                               block.state.Ma(),
                                               30,
                                               cmap='magma_r',
                                               vmax=max_,
                                               vmin=min_)

                    self.realplot.set_aspect('equal')
                    plt.show()
                    plt.pause(0.001)
            ############################################################################################################

            # Increment simulation time
            self.increment_time()
            self.numTimeStep += 1

        if self.inputs.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats('tottime').print_stats()

        print('Date and time: ', datetime.today())

    @staticmethod
    def write_output_nodes(filename: str, array: np.ndarray):
        np.save(file=filename, arr=array)