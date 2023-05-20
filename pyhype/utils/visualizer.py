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

import pathlib
import dataclasses
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from pyhype.solver_config import SolverConfig
from pyhype.states.conservative import ConservativeState

@dataclasses.dataclass
class PlotSetup:
    timesteps: [int]
    size_x: float
    size_y: float
    x_lim: [float]
    y_lim: [float]

class Vizualizer:
    def __init__(self, config: SolverConfig):
        self.config = config
        self.write_path = (
            pathlib.Path(config.write_solution_base) / config.write_solution_name
        )
        self.mesh_path = self.write_path / "mesh"
        self.num_blk = len(list(self.mesh_path.glob('*'))) // 2

        self.cmap = "magma"

    def get_mesh_data(self):
        x_path = self.mesh_path / "mesh_x_blk"
        x = [np.load(f"{x_path}_{blk}.npy")[:, :, 0] for blk in range(self.num_blk)]

        y_path = self.mesh_path / "mesh_y_blk"
        y = [np.load(f"{y_path}_{blk}.npy")[:, :, 0] for blk in range(self.num_blk)]

        return x, y

    def get_solution_file_name(self, blk: int, timestep: int):
        current_path = self.write_path / str(timestep)
        return f"{current_path}\\{self.config.write_solution_name}_blk_{blk}.npy"

    def get_solution_data(self, timestep: int):
        arrays = [np.load(self.get_solution_file_name(blk, timestep)) for blk in range(self.num_blk)]
        states = [ConservativeState(array=array, fluid=self.config.fluid) for array in arrays]
        return states

    def plot(self, setup: PlotSetup):
        x, y = self.get_mesh_data()

        fig, ax1 = plt.subplots(1)
        fig.tight_layout()

        fig.set_size_inches(setup.size_y, setup.size_x)

        ax1.set_title('Density')
        plt.xlim(setup.x_lim)
        plt.ylim(setup.y_lim)

        for timestep in setup.timesteps:
            try:
                states = self.get_solution_data(timestep)

                rho = [state.rho for state in states]
                _min = min([np.amin(v) for v in rho])
                _max = max([np.amax(v) for v in rho])

                for r, x_, y_ in zip(rho, x, y):
                    ax1.contourf(x_, y_, r, 100, cmap='magma', vmin=_min, vmax=_max)

                ax1.set_aspect('equal', adjustable='box')

                write_path = self.write_path / "step_rho"
                plt.savefig(f"{write_path}_{timestep}.png", bbox_inches='tight')
            except Exception as ex:
                print(ex)