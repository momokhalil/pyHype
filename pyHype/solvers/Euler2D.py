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
from pyHype.blocks.base import Blocks
import pyHype.solvers.initial_conditions.initial_conditions as ic

from pyHype.solvers.base import Solver

from typing import TYPE_CHECKING
from typing import Union

if TYPE_CHECKING:
    from pyHype.mesh.base import MeshGenerator

np.set_printoptions(threshold=sys.maxsize)


class Euler2D(Solver):
    def __init__(
        self,
        fvm_type: str = "MUSCL",
        fvm_spatial_order: int = 2,
        fvm_num_quadrature_points: int = 1,
        fvm_gradient_type: str = "GreenGauss",
        fvm_flux_function: str = "Roe",
        fvm_slope_limiter: str = "Venkatakrishnan",
        time_integrator: str = "RK2",
        settings: dict = None,
        mesh_inputs: Union[MeshGenerator, dict] = None,
    ) -> None:

        # --------------------------------------------------------------------------------------------------------------
        # Store mesh features required to create block descriptions

        super().__init__(
            fvm_type=fvm_type,
            fvm_spatial_order=fvm_spatial_order,
            fvm_num_quadrature_points=fvm_num_quadrature_points,
            fvm_gradient_type=fvm_gradient_type,
            fvm_flux_function=fvm_flux_function,
            fvm_slope_limiter=fvm_slope_limiter,
            time_integrator=time_integrator,
            settings=settings,
            mesh_inputs=mesh_inputs,
        )

        # Create Blocks
        print("\t>>> Building solution blocks")
        self._blocks = Blocks(self.inputs)

        print("\n\tSolver Details:\n")
        print(str(self))

        print("\n\tFinished setting up solver")

    def __str__(self):
        __str = (
            "\tA Solver of type Euler2D for solving the 2D Euler\n"
            "\tequations on structured grids using the Finite Volume Method.\n\n"
            "\t"
            + f"{'Finite Volume Method: ':<40} {self.inputs.fvm_type}"
            + "\n"
            + "\t"
            + f"{'Gradient Method: ':<40} {self.inputs.fvm_gradient_type}"
            + "\n"
            + "\t"
            + f"{'Flux Function: ':<40} {self.inputs.fvm_flux_function}"
            + "\n"
            + "\t"
            + f"{'Limiter: ':<40} {self.inputs.fvm_slope_limiter}"
            + "\n"
            + "\t"
            + f"{'Time Integrator: ':<40} {self.inputs.time_integrator}"
        )
        return __str

    def set_IC(self):
        problem_type = self.inputs.problem_type
        if problem_type == "implosion":
            _set_IC = ic.implosion(self.blocks, g=self.inputs.gamma)
        elif problem_type == "explosion":
            _set_IC = ic.explosion(self.blocks, g=self.inputs.gamma)
        elif problem_type == "shockbox":
            _set_IC = ic.shockbox(self.blocks, g=self.inputs.gamma)
        elif problem_type == "supersonic_flood":
            _set_IC = ic.supersonic_flood(self.blocks, g=self.inputs.gamma)
        elif problem_type == "supersonic_rest":
            _set_IC = ic.supersonic_rest(self.blocks, g=self.inputs.gamma)
        elif problem_type == "subsonic_flood":
            _set_IC = ic.subsonic_flood(self.blocks, g=self.inputs.gamma)
        elif problem_type == "subsonic_rest":
            _set_IC = ic.subsonic_rest(self.blocks, g=self.inputs.gamma)
        elif problem_type == "explosion_trapezoid":
            _set_IC = ic.explosion_trapezoid(self.blocks, g=self.inputs.gamma)
        elif problem_type == "explosion_3":
            _set_IC = ic.explosion_3(self.blocks, g=self.inputs.gamma)
        elif problem_type == "mach_reflection":
            _set_IC = ic.mach_reflection(self.blocks, g=self.inputs.gamma)
        else:
            raise ValueError(
                "Initial condition of type "
                + str(self.inputs.problem_type)
                + " has not been specialized."
            )

    def set_BC(self):
        self._blocks.set_BC()

    def solve(self):
        print(
            "\n------------------------------ Initializing Solution Process ---------------------------------"
        )

        print("\nProblem Details: \n")
        for k, v in self._settings_dict.items():
            print("\t" + f"{(str(k) + ': '):<40} {str(v)}")

        print()
        print("\t>>> Setting Initial Conditions")
        self.set_IC()

        print("\t>>> Setting Boundary Conditions")
        self.set_BC()

        """fig, ax = plt.subplots(1)
        for block in self.blocks:
            block.plot(ax=ax)
        plt.show(block=True)"""

        if self.inputs.realplot:
            print("\t>>> Building Real-Time Plot")
            self.build_real_plot()

        if self.inputs.write_solution:
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

        if self.inputs.profile:
            print("\n>>> Enabling Profiler")
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        while self.t < self.t_final:
            if self.numTimeStep % 50 == 0:
                print("\nSimulation time: " + str(self.t / self.inputs.a_inf))
                print("Timestep number: " + str(self.numTimeStep))
            else:
                print(".", end="")

            # Get time step
            self.dt = self.get_dt()
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
        print("Simulation time: " + str(self.t / self.inputs.a_inf))
        print("Timestep number: " + str(self.numTimeStep))
        print()
        print("End of simulation")
        print("Date and time: ", datetime.today())
        print(
            "----------------------------------------------------------------------------------------"
        )
        print()

        if self.inputs.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
            self.profile_data.sort_stats("tottime").print_stats()
