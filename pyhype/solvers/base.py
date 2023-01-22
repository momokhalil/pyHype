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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyhype import execution_prints
from pyhype.blocks.base import BlockDescription
from pyhype.states import ConservativeState

from abc import abstractmethod

from typing import TYPE_CHECKING
from typing import Iterable, Union, Type
from pyhype.mesh.base import MeshGenerator

if TYPE_CHECKING:
    from pyhype.initial_conditions.base import InitialCondition
    from pyhype.blocks.quad_block import QuadBlock
    from pyhype.fluids.base import Fluid
    from pyhype.states import State

np.set_printoptions(threshold=sys.maxsize)


class SolverConfig:
    __slots__ = [
        "fvm_type",
        "fvm_spatial_order",
        "fvm_num_quadrature_points",
        "fvm_gradient_type",
        "fvm_flux_function_type",
        "fvm_slope_limiter_type",
        "time_integrator",
        "mesh",
        "initial_condition",
        "interface_interpolation",
        "reconstruction_type",
        "write_solution",
        "write_solution_mode",
        "write_solution_name",
        "write_every_n_timesteps",
        "plot_every",
        "plot_function",
        "CFL",
        "t_final",
        "realplot",
        "profile",
        "fluid",
        "nx",
        "ny",
        "n",
        "nghost",
        "use_JIT",
    ]

    def __init__(
        self,
        nx: int,
        ny: int,
        CFL: float,
        t_final: float,
        initial_condition: InitialCondition,
        fvm_type: str,
        time_integrator: str,
        fvm_gradient_type: str,
        fvm_flux_function_type: str,
        fvm_slope_limiter_type: str,
        fvm_spatial_order: int,
        fvm_num_quadrature_points: int,
        fluid: Fluid,
        nghost: int = 1,
        use_JIT: bool = True,
        profile: bool = False,
        realplot: bool = False,
        plot_every: int = 20,
        plot_function: str = "Density",
        write_solution: bool = False,
        write_solution_mode: str = "every_n_timesteps",
        write_solution_name: str = "nozzle",
        reconstruction_type: Type[State] = ConservativeState,
        write_every_n_timesteps: int = 40,
        interface_interpolation: str = "arithmetic_average",
        mesh: Union[MeshGenerator, dict] = None,
    ) -> None:

        self.initial_condition = initial_condition

        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.nghost = nghost

        self.CFL = CFL
        self.t_final = t_final

        self.fvm_type = fvm_type
        self.time_integrator = time_integrator
        self.fvm_gradient_type = fvm_gradient_type
        self.fvm_flux_function_type = fvm_flux_function_type
        self.fvm_slope_limiter_type = fvm_slope_limiter_type
        self.fvm_spatial_order = fvm_spatial_order
        self.fvm_num_quadrature_points = fvm_num_quadrature_points

        self.reconstruction_type = reconstruction_type
        self.interface_interpolation = interface_interpolation

        self.fluid = fluid

        self.use_JIT = use_JIT
        self.profile = profile
        self.realplot = realplot
        self.plot_every = plot_every
        self.plot_function = plot_function

        self.write_solution = write_solution
        self.write_solution_mode = write_solution_mode
        self.write_solution_name = write_solution_name
        self.write_every_n_timesteps = write_every_n_timesteps

        self.mesh = None
        self.set_mesh_config(mesh)

    def __str__(self):
        return "".join(f"\t{atr}: {getattr(self, atr)}\n" for atr in self.__slots__)

    def set_mesh_config(self, mesh: Union[MeshGenerator, dict] = None) -> None:
        mesh_info = mesh.dict if isinstance(mesh, MeshGenerator) else mesh
        self.mesh = {
            blk_num: BlockDescription(
                nx=self.nx,
                ny=self.ny,
                blk_input=blk_data,
                nghost=self.nghost,
            )
            for (blk_num, blk_data) in mesh_info.items()
        }


class Solver:
    def __init__(
        self,
        config: SolverConfig,
    ) -> None:

        print(execution_prints.PYHYPE)
        print(execution_prints.LICENSE)
        print(
            "\n------------------------------------ Setting-Up Solver ---------------------------------------\n"
        )

        self.config = config
        self.fluid = config.fluid
        self.cmap = LinearSegmentedColormap.from_list(
            "my_map", ["royalblue", "midnightblue", "black"]
        )

        print("\t>>> Initializing basic solution attributes")
        self.t = 0
        self.dt = 0
        self.numTimeStep = 0
        self.CFL = self.config.CFL
        self.t_final = self.config.t_final * self.fluid.far_field.a
        self.profile_data = None
        self.realfig, self.realplot = None, None
        self.plot = None
        self._blocks = None

    @abstractmethod
    def apply_initial_condition(self):
        raise NotImplementedError

    @abstractmethod
    def apply_boundary_condition(self):
        raise NotImplementedError

    @property
    def blocks(self) -> Iterable[QuadBlock]:
        return self._blocks.blocks.values()

    def get_dt(self):
        """
        Return the time step for all blocks handled by this process based on the CFL condition.

        Parameters:
            - None

        Returns:
            - dt (np.float): Float representing the value of the time step
        """
        dt = min([block.get_dt() for block in self.blocks])
        return self.t_final - self.t if self.t_final - self.t < dt else dt

    def increment_time(self):
        self.t += self.dt

    @staticmethod
    def write_output_nodes(filename: str, array: np.ndarray):
        np.save(file=filename, arr=array)

    def write_solution(self):
        if self.config.write_solution_mode == "every_n_timesteps":
            if self.numTimeStep % self.config.write_every_n_timesteps == 0:
                for block in self.blocks:
                    self.write_output_nodes(
                        "./"
                        + self.config.write_solution_name
                        + "_"
                        + str(self.numTimeStep)
                        + "_blk_"
                        + str(block.global_nBLK),
                        block.state.data,
                    )

    def plot_func_selector(self, state) -> np.ndarray:
        """
        Evaluates a function based on the solution data in state for plotting.
        :param state: State object to plot
        :return: Array of the evaluation function
        """
        if self.config.plot_function == "Mach Number":
            return state.Ma()
        if self.config.plot_function == "Density":
            return state.rho
        if self.config.plot_function == "X velocity":
            return state.u
        if self.config.plot_function == "Y velocity":
            return state.v
        if self.config.plot_function == "Pressure":
            return state.p
        if self.config.plot_function == "Energy":
            return state.e
        # Default to density
        return state.rho

    def real_plot(self):
        if self.numTimeStep % self.config.plot_every == 0:
            data = [
                (
                    block.mesh.x[:, :, 0],
                    block.mesh.y[:, :, 0],
                    self.plot_func_selector(block.state),
                )
                for block in self.blocks
            ]

            for (x, y, z) in data:
                self.realplot.contourf(
                    x,
                    y,
                    z,
                    50,
                    cmap="magma",
                    vmax=max([np.max(v[2]) for v in data]),
                    vmin=min([np.min(v[2]) for v in data]),
                )
            self.realplot.set_aspect("equal")
            plt.show()
            plt.pause(0.001)
            plt.cla()

    def build_real_plot(self):
        plt.ion()
        self.realfig, self.realplot = plt.subplots(1)

        blks = self.config.mesh.values()

        sw_x = min([blk.geometry.vertices.SW[0] for blk in blks])
        nw_x = min([blk.geometry.vertices.NW[0] for blk in blks])
        sw_y = min([blk.geometry.vertices.SW[1] for blk in blks])
        se_y = min([blk.geometry.vertices.SE[1] for blk in blks])

        se_x = max([blk.geometry.vertices.SE[0] for blk in blks])
        ne_x = max([blk.geometry.vertices.NE[0] for blk in blks])
        nw_y = max([blk.geometry.vertices.NW[1] for blk in blks])
        ne_y = max([blk.geometry.vertices.NE[1] for blk in blks])

        W = max(se_x, ne_x) - min(sw_x, nw_x)
        L = max(nw_y, ne_y) - min(sw_y, se_y)

        w = 6

        self.realplot.figure.set_size_inches(w, w * (L / W))
