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
import logging
import pstats
import cProfile
import numpy as np
from datetime import datetime
from pyhype.blocks.base import Blocks
from pyhype.solvers.base import Solver
from pyhype.solver_config import SolverConfig

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.mesh.base import MeshGenerator

np.set_printoptions(threshold=sys.maxsize)
logging.basicConfig(level=logging.INFO)


class Euler2D(Solver):
    def __init__(
        self,
        config: SolverConfig,
        mesh_config: Union[MeshGenerator, dict],
    ) -> None:

        super().__init__(config=config, mesh_config=mesh_config)

        self._logger.info("\t>>> Building solution blocks")
        self._blocks = Blocks(
            config=self.config,
            mesh_config=self.mesh_config,
        )

        if self.cpu == 0:
            self._logger.info("\n\tSolver Details:\n")
            self._logger.info(self)
            self._logger.info("\n\tFinished setting up solver")

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
        if self.cpu == 0:
            self._logger.info(
                "\n------------------------------ Initializing Solution Process ---------------------------------"
            )

        if self.cpu == 0:
            self._logger.info("\nProblem Details: \n")
            self._logger.info(self.config)

        if self.cpu == 0:
            self._logger.info("\t>>> Setting Initial Conditions")
        self.apply_initial_condition()

        if self.cpu == 0:
            self._logger.info("\t>>> Setting Boundary Conditions")
        self.apply_boundary_condition()

        if self.config.realplot:
            self._logger.info("\t>>> Building Real-Time Plot")
            self.build_real_plot()

        if self.config.write_solution:
            if self.cpu == 0:
                self._logger.info("\t>>> Writing Mesh to File")
            for block in self.blocks:
                self.write_output_nodes(
                    "./mesh_blk_x_" + str(block.global_nBLK), block.mesh.x
                )
                self.write_output_nodes(
                    "./mesh_blk_y_" + str(block.global_nBLK), block.mesh.y
                )

        if self.cpu == 0:
            self._logger.info(
                "\n------------------------------------- Start Simulation ---------------------------------------\n"
            )
            self._logger.info(f"Date and time: {datetime.today()}")

        if self.config.profile:
            if self.cpu == 0:
                self._logger.info("\n>>> Enabling Profiler")
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        while self.t < self.t_final:
            if self.cpu == 0 and self.num_time_step % 50 == 0:
                self._logger.info(
                    f"Simulation time: {str(self.t / self.fluid.far_field.a)}, Timestep number: {str(self.num_time_step)}",
                )

            # Get time step
            self.mpi.Barrier()
            dt = self.get_dt()
            self._blocks.update(dt)

            if self.config.write_solution:
                self.write_solution()

            self.t += dt
            self.num_time_step += 1

        if self.cpu == 0:
            self._logger.info(
                f"Simulation time: {str(self.t / self.fluid.far_field.a)}, Timestep number: {str(self.num_time_step)}",
            )
            self._logger.info("End of simulation")
            self._logger.info(f"Date and time: {datetime.today()}")
            self._logger.info(
                "----------------------------------------------------------------------------------------"
            )

        if self.config.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats("tottime").print_stats()
