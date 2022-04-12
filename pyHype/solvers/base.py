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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pyHype import execution_prints
from pyHype.mesh.base import BlockDescription

from abc import abstractmethod

from typing import TYPE_CHECKING
from typing import Iterable, Union
from pyHype.mesh.base import MeshGenerator

if TYPE_CHECKING:
    from pyHype.blocks.QuadBlock import QuadBlock

np.set_printoptions(threshold=sys.maxsize)


class ProblemInput:
    __REQUIRED__ = ['problem_type',
                    'interface_interpolation',
                    'reconstruction_type',
                    'write_solution',
                    'write_solution_mode',
                    'write_solution_name',
                    'write_every_n_timesteps',
                    'plot_every',
                    'CFL',
                    't_final',
                    'realplot',
                    'profile',
                    'gamma',
                    'rho_inf',
                    'a_inf',
                    'R',
                    'nx',
                    'ny',
                    'nghost'
                    ]


    def __init__(self,
                 fvm_type:                  str = 'MUSCL',
                 fvm_spatial_order:         int = 2,
                 fvm_num_quadrature_points: int = 1,
                 fvm_gradient_type:         str = 'GreenGauss',
                 fvm_flux_function:         str = 'Roe',
                 fvm_slope_limiter:         str = 'Venkatakrishnan',
                 time_integrator:           str = 'RK2',
                 settings:                  dict = None,
                 mesh_inputs:               Union[MeshGenerator, dict] = None,
                 ) -> None:
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # Check input dictionary to check if all required fields are present
        self._check_input_settings(settings)

        # REQUIRED

        self.fvm_type = fvm_type
        self.fvm_spatial_order = fvm_spatial_order
        self.fvm_num_quadrature_points = fvm_num_quadrature_points
        self.fvm_gradient_type = fvm_gradient_type
        self.fvm_flux_function = fvm_flux_function
        self.fvm_slope_limiter = fvm_slope_limiter
        self.time_integrator = time_integrator
        self.n = settings['nx'] * settings['ny']
        self.mesh_inputs = mesh_inputs

        # Set all input parameters
        for key, val in settings.items():
            self.__setattr__(key, val)


    def _check_input_settings(self, input_dict: dict) -> None:
        for key in self.__REQUIRED__:
            if key not in input_dict.keys():
                raise KeyError(key + ' not found in inputs.')


class Solver:
    def __init__(self,
                 fvm_type:                  str = 'MUSCL',
                 fvm_spatial_order:         int = 2,
                 fvm_num_quadrature_points: int = 1,
                 fvm_gradient_type:         str = 'GreenGauss',
                 fvm_flux_function:         str = 'Roe',
                 fvm_slope_limiter:         str = 'Venkatakrishnan',
                 time_integrator:           str = 'RK2',
                 settings:                  dict = None,
                 mesh_inputs:               Union[MeshGenerator, dict] = None,
                 ) -> None:

        print(execution_prints.pyhype)
        print(execution_prints.lice)
        print('\n------------------------------------ Setting-Up Solver ---------------------------------------\n')

        self._settings_dict = settings

        print('\t>>> Building Mesh Descriptors')

        _mesh = mesh_inputs.dict if isinstance(mesh_inputs, MeshGenerator) else mesh_inputs
        _mesh_inputs = {blk: BlockDescription(blkData,
                                              nx=settings['nx'],
                                              ny=settings['ny'],
                                              nghost=settings['nghost']) for (blk, blkData) in _mesh.items()}
        self.cmap = LinearSegmentedColormap.from_list('my_map', ['royalblue', 'midnightblue', 'black'])

        print('\t>>> Checking all boundary condition types')
        _bc_type_names = ['BCTypeE', 'BCTypeW', 'BCTypeN', 'BCTypeS']
        self._all_BC_types = []
        for blkdata in _mesh.values():
            for bc_name in _bc_type_names:
                if blkdata[bc_name] not in self._all_BC_types:
                    self._all_BC_types.append(blkdata[bc_name])

        print('\t>>> Building Settings Descriptors')
        self.inputs = ProblemInput(fvm_type=fvm_type,
                                   fvm_spatial_order=fvm_spatial_order,
                                   fvm_num_quadrature_points=fvm_num_quadrature_points,
                                   fvm_gradient_type=fvm_gradient_type,
                                   fvm_flux_function=fvm_flux_function,
                                   fvm_slope_limiter=fvm_slope_limiter,
                                   time_integrator=time_integrator,
                                   settings=settings,
                                   mesh_inputs=_mesh_inputs)

        print('\t>>> Initializing basic solution attributes')
        self.t = 0
        self.dt = 0
        self.numTimeStep = 0
        self.CFL = self.inputs.CFL
        self.t_final = self.inputs.t_final * self.inputs.a_inf
        self.profile_data = None
        self.realfig, self.realplot = None, None
        self.plot = None
        self._blocks = None

    @abstractmethod
    def set_IC(self):
        raise NotImplementedError

    @abstractmethod
    def set_BC(self):
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
        _dt = min([block.get_dt() for block in self.blocks])
        return self.t_final - self.t if self.t_final - self.t < _dt else _dt


    def increment_time(self):
        self.t += self.dt

    @staticmethod
    def write_output_nodes(filename: str, array: np.ndarray):
        np.save(file=filename, arr=array)

    def write_solution(self):
        if self.inputs.write_solution_mode == 'every_n_timesteps':
            if self.numTimeStep % self.inputs.write_every_n_timesteps == 0:
                for block in self.blocks:
                    self.write_output_nodes('./' + self.inputs.write_solution_name
                                                 + '_' + str(self.numTimeStep)
                                                 + '_blk_' + str(block.global_nBLK),
                                            block.state.U)

    def real_plot(self):
        if self.numTimeStep % self.inputs.plot_every == 0:
            data = [(block.mesh.x[:, :, 0], block.mesh.y[:, :, 0], block.state.rho)  for block in self.blocks]
            for _vars in data:
                self.realplot.contourf(*_vars, 50, cmap='magma',
                                       vmax=max([np.max(v[2]) for v in data]),
                                       vmin=min([np.min(v[2]) for v in data]))
            self.realplot.set_aspect('equal')
            plt.show()
            plt.pause(0.001)
            plt.cla()

    def build_real_plot(self):
        plt.ion()
        self.realfig, self.realplot = plt.subplots(1)

        blks = list(self.inputs.mesh_inputs.values())

        sw_x = min([blk.SW[0] for blk in blks])
        nw_x = min([blk.NW[0] for blk in blks])
        sw_y = min([blk.SW[1] for blk in blks])
        se_y = min([blk.SE[1] for blk in blks])

        se_x = max([blk.SE[0] for blk in blks])
        ne_x = max([blk.NE[0] for blk in blks])
        nw_y = max([blk.NW[1] for blk in blks])
        ne_y = max([blk.NE[1] for blk in blks])

        W = max(se_x, ne_x) - min(sw_x, nw_x)
        L = max(nw_y, ne_y) - min(sw_y, se_y)

        w = 6

        self.realplot.figure.set_size_inches(w, w * (L / W))
