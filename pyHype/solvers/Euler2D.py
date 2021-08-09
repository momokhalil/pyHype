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
from pyHype.blocks.base import Blocks
import pyHype.solvers.initial_conditions.initial_conditions as ic

from pyHype.solvers.base import Solver, _DEFINED_IC_

np.set_printoptions(threshold=sys.maxsize)


class Euler2D(Solver):
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

        super().__init__(fvm=fvm, gradient=gradient, flux_function=flux_function, limiter=limiter,
                                   integrator=integrator, settings=settings)

        # Create Blocks
        self._blocks = Blocks(self.inputs)

    def __str__(self):
        __str = '\tA Solver of type Euler2D for solving the 2D Euler\n ' \
                '\tequations on structured grids using the Finite Volume Method.\n\n' \
                '\tSolver Details:\n' + \
                '\t--------------\n' + \
                '\t' + f"{'Finite Volume Method: ':<35} {self.inputs.fvm}" + '\n' + \
                '\t' + f"{'Gradient Method: ':<35} {self.inputs.gradient}" + '\n' + \
                '\t' + f"{'Flux Function: ':<35} {self.inputs.flux_function}" + '\n' + \
                '\t' + f"{'Limiter: ':<35} {self.inputs.limiter}" + '\n' + \
                '\t' + f"{'Time Integrator: ':<35} {self.inputs.integrator}" + '\n'
        return __str


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

    def solve(self):
        print('\nProblem Details: \n'
              '--------------\n')
        for k, v in self._settings_dict.items():
            print('\t' + f"{(str(k) + ': '):<40} {str(v)}")

        print('\nSolver Description:\n'
              '--------------------\n')
        print(str(self))


        print()
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
            self.build_real_plot()

        if self.inputs.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        if self.inputs.write_solution:
            for block in self.blocks:
                self.write_output_nodes('./mesh_blk_x_' + str(block.global_nBLK), block.mesh.x)
                self.write_output_nodes('./mesh_blk_y_' + str(block.global_nBLK), block.mesh.y)


        print('Start simulation')
        while self.t < self.t_final:

            if self.numTimeStep % 50 == 0:
                print()
                print('Simulation time: ' + str(self.t / self.inputs.a_inf))
                print('Timestep number: ' + str(self.numTimeStep))
            else:
                print('.', end='')

            self.dt = self.get_dt()

            # print('update block')
            self._blocks.update(self.dt)

            ############################################################################################################
            # THIS IS FOR DEBUGGING PURPOSES ONLY
            if self.inputs.write_solution:
                self.write_solution()

            if self.inputs.realplot:
                self.real_plot()
            ############################################################################################################

            # Increment simulation time
            self.increment_time()
            self.numTimeStep += 1

        print()
        print()
        print('Simulation time: ' + str(self.t / self.inputs.a_inf))
        print('Timestep number: ' + str(self.numTimeStep))
        print()
        print('End of simulation')
        print('Date and time: ', datetime.today())
        print('----------------------------------------------------------------------------------------')
        print()

        if self.inputs.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats('tottime').print_stats()
