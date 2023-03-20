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

import logging
logging.basicConfig(level=logging.INFO)

from typing import Optional, Any

import mpi4py as mpi
from pyhype.solver_config import SolverConfig

class Logger:
    def __init__(self, config: SolverConfig, name: Optional[str] = None):
        self._config = config
        self._name = name if name is not None else self.__class__.__name__
        self._logger = logging.getLogger(self._name)
        self._mpi = mpi.MPI.COMM_WORLD
        self._cpu = self._mpi.Get_rank()

        if config.show_log_for_procs == "all":
            self._show_logs = True
        else:
            self._show_logs = self._cpu in config.show_log_for_procs

    @property
    def name(self):
        return self._name

    def info(self, message: Any):
        if self._show_logs:
            self._logger.info(message)

    def debug(self, message: Any):
        if self._show_logs:
            self._logger.debug(message)

    def warning(self, message: Any):
        if self._show_logs:
            self._logger.warning(message)

    def error(self, message: Any):
        if self._show_logs:
            self._logger.error(message)
