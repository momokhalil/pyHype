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

import mpi4py.MPI

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
from pyhype.states.base import RealizabilityException
from pyhype.time_marching.factory import TimeIntegratorFactory

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

        self.mpi.Barrier()
        self._blocks = Blocks(
            config=self.config,
            mesh_config=self.mesh_config,
        )
        self._time_integrator = TimeIntegratorFactory.create(config=config)

        self.mpi.Barrier()
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
        self._pre_process_solve()
        self._solve()
        self._post_process_solve()

    def _pre_process_solve(self):
        self._logger.info(
            "\n------------------------------ Initializing Solution Process ---------------------------------"
        )

        self._logger.info("\nProblem Details: \n")
        self._logger.info(self.config)

        self._logger.info("\t>>> Setting Initial Conditions")
        self.mpi.Barrier()
        self.apply_initial_condition()

        self._logger.info("\t>>> Setting Boundary Conditions")
        self.mpi.Barrier()
        self.apply_boundary_condition()

        if self.config.realplot:
            self._logger.info("\t>>> Building Real-Time Plot")
            self.build_real_plot()

        if self.config.write_solution:
            self._logger.info("\t>>> Writing Mesh to File")

            self.write_mesh()

        self._logger.info(
            "\n------------------------------------- Start Simulation ---------------------------------------\n"
        )
        self._logger.info(f"Date and time: {datetime.today()}")

    def _post_process_solve(self):
        self._logger.info(
            f"Simulation time: {str(self.t / self.fluid.far_field.a)}, Timestep number: {str(self.num_time_step)}",
        )
        self._logger.info("End of simulation")
        self._logger.info(f"Date and time: {datetime.today()}")
        self._logger.info(
            "----------------------------------------------------------------------------------------"
        )

        mpi4py.MPI.Finalize()

    def _realizability_check(self):
        realizable = self._blocks.realizable()
        failed = [
            fail for fail in realizable if isinstance(fail, RealizabilityException)
        ]
        if len(failed):
            for fail in failed:
                self._logger.error(fail)
            self.mpi.Abort()

    def _log_simulation_progress(self) -> None:
        """
        Logs the current time step and the number of performed time steps to console.

        :return: None
        """
        t = self.t / self.fluid.far_field.a
        self._logger.info(
            f"Simulation time: {t}, Timestep number: {self.num_time_step}",
        )

    def _update_solution_blocks(self, dt: float) -> None:
        """
        Calls the time marching operator to update the solution state in all
        solution blocks.

        :param dt: current time step
        :return: None
        """
        self._time_integrator.integrate(dt, self._blocks)

    def _solve(self) -> None:
        """
        Executes the solution procedure for solving the 2D Euler equations in
        the given domain with the given simulation parameters. It iterates through
        time and updates the solution blocks at each time iteration, ensuring the
        time step obeys the CFL condition for inviscid flow to maintain a stable
        numerical scheme. It may also write the solution state to file at given
        intervals if the user wishes.

        :return: None
        """

        self.mpi.Barrier()

        profiler = None
        if self.config.profile:
            self._logger.info("\n>>> Enabling Profiler")
            profiler = cProfile.Profile()
            profiler.enable()

        while self.t < self.t_final:
            if self.num_time_step % 50 == 0:
                self._log_simulation_progress()

            self.mpi.Barrier()
            dt = self.get_dt()

            if len(self._blocks):
                self._update_solution_blocks(dt=dt)
                self._realizability_check()

            if self.config.write_solution:
                self.write_solution()

            self.t += dt
            self.num_time_step += 1

        if self.config.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            if self.cpu == 0:
                self.profile_data.sort_stats("tottime").print_stats(50)
