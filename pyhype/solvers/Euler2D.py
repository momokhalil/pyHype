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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import sys
import pstats
import cProfile
import numpy as np
from datetime import datetime
from pyhype.factory import Factory
from pyhype.blocks.base import Blocks
from pyhype.solvers.base import Solver
from pyhype.solver_config import SolverConfig

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.mesh.base import MeshGenerator

np.set_printoptions(threshold=sys.maxsize)


class Euler2D(Solver):
    def __init__(
        self,
        config: SolverConfig,
        mesh_config: Union[MeshGenerator, dict],
    ) -> None:

        super().__init__(config=config, mesh_config=mesh_config)

        print("\t>>> Building solution blocks")
        self._blocks = Blocks(
            config=self.config,
            mesh_config=self.mesh_config,
        )

        print("\n\tSolver Details:\n")
        print(self)

        print("\n\tFinished setting up solver")

    def __str__(self):
        string = (
            "\tA Solver of type Euler2D for solving the 2D Euler\n"
            "\tequations on structured grids using the Finite Volume Method.\n\n"
            "\t"
            + f"{'Finite Volume Method: ':<40} {self.config.fvm_type}"
            + "\n"
            + "\t"
            + f"{'Gradient Method: ':<40} {self.config.fvm_gradient_type}"
            + "\n"
            + "\t"
            + f"{'Flux Function: ':<40} {self.config.fvm_flux_function_type}"
            + "\n"
            + "\t"
            + f"{'Limiter: ':<40} {self.config.fvm_slope_limiter_type}"
            + "\n"
            + "\t"
            + f"{'Time Integrator: ':<40} {self.config.time_integrator}"
        )
        return string

    def apply_initial_condition(self):
        for block in self.blocks:
            self.config.initial_condition.apply_to_block(block)

    def apply_boundary_condition(self):
        self._blocks.apply_boundary_condition()

    def solve(self):
        print(
            "\n------------------------------ Initializing Solution Process ---------------------------------"
        )

        print("\nProblem Details: \n")
        print(self.config)

        print()
        print("\t>>> Setting Initial Conditions")
        self.apply_initial_condition()

        print("\t>>> Setting Boundary Conditions")
        self.apply_boundary_condition()

        if self.config.realplot:
            print("\t>>> Building Real-Time Plot")
            self.build_real_plot()

        if self.config.write_solution:
            print("\t>>> Writing Mesh to File")
            for block in self.blocks:
                self.write_output_nodes(
                    "./mesh_blk_x_" + str(block.global_nBLK), block.mesh.x
                )
                self.write_output_nodes(
                    "./mesh_blk_y_" + str(block.global_nBLK), block.mesh.y
                )

        print(
            "\n------------------------------------- Start Simulation ---------------------------------------\n"
        )
        print("Date and time: ", datetime.today())

        if self.config.profile:
            print("\n>>> Enabling Profiler")
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        while self.t < self.t_final:
            if self.numTimeStep % 50 == 0:
                print("\nSimulation time: " + str(self.t / self.fluid.far_field.a))
                print("Timestep number: " + str(self.numTimeStep))
            else:
                print(".", end="")

            # Get time step
            self.dt = self.get_dt()
            self._blocks.update(self.dt)

            ############################################################################################################
            # THIS IS FOR DEBUGGING PURPOSES ONLY
            if self.config.write_solution:
                self.write_solution()

            if self.config.realplot:
                self.real_plot()
            ############################################################################################################

            # Increment simulation time
            self.increment_time()
            self.numTimeStep += 1

        print()
        print()
        print("Simulation time: " + str(self.t / self.config.fluid.far_field.a))
        print("Timestep number: " + str(self.numTimeStep))
        print()
        print("End of simulation")
        print("Date and time: ", datetime.today())
        print(
            "----------------------------------------------------------------------------------------"
        )
        print()

        if self.config.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats("tottime").print_stats()
