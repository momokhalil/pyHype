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
from pyhype.mesh.base import MeshGenerator

from abc import abstractmethod

from typing import Iterable, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pyhype.solver_config import SolverConfig
    from pyhype.blocks.quad_block import QuadBlock

np.set_printoptions(threshold=sys.maxsize)


class Solver:
    def __init__(
        self,
        config: SolverConfig,
        mesh_config: Union[MeshGenerator, dict],
    ) -> None:

        print(execution_prints.PYHYPE)
        print(execution_prints.LICENSE)
        print(
            "\n------------------------------------ Setting-Up Solver ---------------------------------------\n"
        )

        self.config = config
        self.mesh_config = self.get_mesh_config(mesh_config)

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

    def get_mesh_config(self, mesh: Union[MeshGenerator, dict]):
        mesh_info = mesh.dict if isinstance(mesh, MeshGenerator) else mesh
        return {
            blk_num: BlockDescription(
                nx=self.config.nx,
                ny=self.config.ny,
                blk_input=blk_data,
                nghost=self.config.nghost,
            )
            for (blk_num, blk_data) in mesh_info.items()
        }

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

        blks = self.mesh_config.values()

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
