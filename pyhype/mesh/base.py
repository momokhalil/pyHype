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
import os
from abc import abstractmethod

from pyhype.utils.utils import (
    CornerPropertyContainer,
    FullPropertyContainer,
)

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class BlockGeometry:
    def __init__(
        self,
        NE: [float] = None,
        NW: [float] = None,
        SE: [float] = None,
        SW: [float] = None,
        nx: int = None,
        ny: int = None,
        nghost: int = None,
    ):
        self.vertices = CornerPropertyContainer(NE=NE, NW=NW, SE=SE, SW=SW)
        self.n = nx * ny
        self.nx = nx
        self.ny = ny
        self.nghost = nghost


class BlockInfo:
    def __init__(self, blk_input: dict):
        # Set parameter attributes from input dict
        self.nBLK = blk_input["nBLK"]

        self.neighbors = FullPropertyContainer(
            E=blk_input["NeighborE"],
            W=blk_input["NeighborW"],
            N=blk_input["NeighborN"],
            S=blk_input["NeighborS"],
            NE=blk_input["NeighborNE"],
            NW=blk_input["NeighborNW"],
            SE=blk_input["NeighborSE"],
            SW=blk_input["NeighborSW"],
        )
        self.bc = FullPropertyContainer(
            E=blk_input["BCTypeE"],
            W=blk_input["BCTypeW"],
            N=blk_input["BCTypeN"],
            S=blk_input["BCTypeS"],
            NE=blk_input["BCTypeNE"],
            NW=blk_input["BCTypeNW"],
            SE=blk_input["BCTypeSE"],
            SW=blk_input["BCTypeSW"],
        )


class BlockDescription:
    def __init__(self, nx: int, ny: int, blk_input: dict = None, nghost: int = None):
        # Set parameter attributes from input dict
        self.info = None
        if blk_input is not None:
            self.info = BlockInfo(blk_input=blk_input)

        self.geometry = BlockGeometry(
            NE=blk_input["NE"],
            NW=blk_input["NW"],
            SE=blk_input["SE"],
            SW=blk_input["SW"],
            nx=nx,
            ny=ny,
            nghost=nghost,
        )

    @property
    def neighbors(self):
        return self.info.neighbors

    @property
    def bc(self):
        return self.info.bc


class GridLocation:
    def __init__(self, x: np.ndarray = None, y: np.ndarray = None):
        self.x = x
        self.y = y


class CellFace:
    def __init__(self) -> None:

        # Define face midpoint locations
        self.xmid = None
        self.ymid = None

        # Define normals
        self.xnorm = None
        self.ynorm = None

        # Define angles
        self.theta = None

        # Define face length
        self.L = None


class _mesh_transfinite_gen:
    def __init__(self) -> None:
        self.x = None
        self.y = None
        self.nx = None
        self.ny = None

    def _get_interior_mesh_transfinite(self, x: np.ndarray, y: np.ndarray):

        _im = np.linspace(1 / self.ny, (self.ny - 1) / self.ny, self.ny - 2)
        _jm = np.linspace(1 / self.nx, (self.nx - 1) / self.nx, self.nx - 2)

        jm, im = np.meshgrid(_jm, _im)

        _mi = np.linspace((self.ny - 1) / self.ny, 1 / self.ny, self.ny - 2)
        _mj = np.linspace((self.nx - 1) / self.nx, 1 / self.nx, self.nx - 2)

        mj, mi = np.meshgrid(_mj, _mi)

        x[1:-1, 1:-1] = self.__get_kernel_transfinite(x, im, jm, mi, mj)
        y[1:-1, 1:-1] = self.__get_kernel_transfinite(y, im, jm, mi, mj)

        return x, y

    @staticmethod
    def __get_kernel_transfinite(
        x: np.ndarray,
        im: np.ndarray,
        jm: np.ndarray,
        mi: np.ndarray,
        mj: np.ndarray,
    ):

        return (
            mi * x[0, 1:-1]
            + im * x[-1, 1:-1]
            + mj * x[1:-1, 0, None]
            + jm * x[1:-1, -1, None]
            - mi * mj * x[0, 0]
            - mi * jm * x[0, -1]
            - im * mj * x[-1, 0]
            - im * jm * x[-1, -1]
        )


class MeshGenerator:
    def __init__(self):
        self.dict = {}

    @abstractmethod
    def _create_block_descriptions(self):
        raise NotImplementedError


class QuadMeshGenerator(MeshGenerator, _mesh_transfinite_gen):
    def __init__(
        self,
        nx_blk: int,
        ny_blk: int,
        BCE: [str],
        BCW: [str],
        BCN: [str],
        BCS: [str],
        NE: [float] = None,
        NW: [float] = None,
        SE: [float] = None,
        SW: [float] = None,
        left_x: [float] = None,
        left_y: [float] = None,
        right_x: [float] = None,
        right_y: [float] = None,
        top_x: [float] = None,
        top_y: [float] = None,
        bot_x: [float] = None,
        bot_y: [float] = None,
        blk_num_offset: int = 0,
    ) -> None:

        super().__init__()

        self.nx, self.ny = nx_blk + 1, ny_blk + 1
        self._blk_num_offset = blk_num_offset

        self.BCE = [BCE for _ in range(ny_blk)] if isinstance(BCE, str) else BCE
        self.BCW = [BCW for _ in range(ny_blk)] if isinstance(BCW, str) else BCW
        self.BCN = [BCN for _ in range(nx_blk)] if isinstance(BCN, str) else BCN
        self.BCS = [BCS for _ in range(nx_blk)] if isinstance(BCS, str) else BCS

        _x, _y = np.meshgrid(np.linspace(0, 1, self.nx), np.linspace(0, 1, self.ny))

        _x[0, :] = np.linspace(SW[0], SE[0], self.nx) if bot_x is None else bot_x
        _y[0, :] = np.linspace(SW[1], SE[1], self.nx) if bot_y is None else bot_y
        _x[-1, :] = np.linspace(NW[0], NE[0], self.nx) if top_x is None else top_x
        _y[-1, :] = np.linspace(NW[1], NE[1], self.nx) if top_y is None else top_y
        _x[:, 0] = np.linspace(SW[0], NW[0], self.ny) if left_x is None else left_x
        _y[:, 0] = np.linspace(SW[1], NW[1], self.ny) if left_y is None else left_y
        _x[:, -1] = np.linspace(SE[0], NE[0], self.ny) if right_x is None else right_x
        _y[:, -1] = np.linspace(SE[1], NE[1], self.ny) if right_y is None else right_y

        self.x, self.y = self._get_interior_mesh_transfinite(_x, _y)
        self.dict = self._create_block_descriptions()

    def _create_block_descriptions(self):

        _ny = self.ny - 1
        _nx = self.nx - 1

        dicts = {}

        for i in range(_ny):
            for j in range(_nx):
                _num = _nx * i + j + self._blk_num_offset
                _blk = {
                    "nBLK": _num,
                    "NW": [self.x[i + 1, j], self.y[i + 1, j]],
                    "NE": [self.x[i + 1, j + 1], self.y[i + 1, j + 1]],
                    "SW": [self.x[i, j], self.y[i, j]],
                    "SE": [self.x[i, j + 1], self.y[i, j + 1]],
                    "NeighborE": _num + 1 if j < _nx - 1 else None,
                    "NeighborW": _num - 1 if j > 0 else None,
                    "NeighborN": _num + _nx if _num + _nx < _ny * _nx else None,
                    "NeighborS": _num - _nx if _num - _nx >= 0 else None,
                    "BCTypeE": self.BCE[i] if j == _nx - 1 else "None",
                    "BCTypeW": self.BCW[i] if j == 0 else "None",
                    "BCTypeN": self.BCN[j] if i == _ny - 1 else "None",
                    "BCTypeS": self.BCS[j] if i == 0 else "None",
                }
                dicts[_num] = _blk

        return dicts
