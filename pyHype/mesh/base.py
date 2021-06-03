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

import numpy as np
from typing import Union


class BlockDescription:
    def __init__(self, blk_input):

        # Set parameter attributes from input dict
        self.nBLK = blk_input['nBLK']
        self.n = blk_input['n']
        self.nx = blk_input['nx']
        self.ny = blk_input['ny']
        self.NeighborE = blk_input['NeighborE']
        self.NeighborW = blk_input['NeighborW']
        self.NeighborN = blk_input['NeighborN']
        self.NeighborS = blk_input['NeighborS']
        self.NE = blk_input['NE']
        self.NW = blk_input['NW']
        self.SE = blk_input['SE']
        self.SW = blk_input['SW']
        self.BCTypeE = blk_input['BCTypeE']
        self.BCTypeW = blk_input['BCTypeW']
        self.BCTypeN = blk_input['BCTypeN']
        self.BCTypeS = blk_input['BCTypeS']


class Mesh:
    def __init__(self, inputs, mesh_data):

        self.nx = inputs.nx
        self.ny = inputs.ny
        self.nghost = inputs.nghost

        self.inputs = inputs

        self.vertices = Vertices(NW=mesh_data.NW,
                                 NE=mesh_data.NE,
                                 SW=mesh_data.SW,
                                 SE=mesh_data.SE)

        self.x = np.zeros((self.ny, self.nx))
        self.y = np.zeros((self.ny, self.nx))

        self.xc = np.zeros((self.ny + 1, self.nx + 1))
        self.yc = np.zeros((self.ny + 1, self.nx + 1))

        self.A = np.zeros((self.ny, self.nx))

        self.create_mesh()

        self.Lx    = self.vertices.NE[0] - self.vertices.NW[0]
        self.Ly    = self.vertices.NE[1] - self.vertices.SE[1]

        self.dx     = self.Lx / (self.nx + 1)
        self.dy     = self.Ly / (self.ny + 1)

        self.normx = np.zeros((self.ny + 1, self.nx + 1))
        self.normy = np.zeros((self.ny + 1, self.nx + 1))

    def create_mesh(self):

        # East edge x and y node locations
        Ex = np.linspace(self.vertices.SE[0], self.vertices.NE[0], self.ny)
        Ey = np.linspace(self.vertices.SE[1], self.vertices.NE[1], self.ny)
        # West edge x and y node locations
        Wx = np.linspace(self.vertices.SW[0], self.vertices.NW[0], self.ny)
        Wy = np.linspace(self.vertices.SW[1], self.vertices.NW[1], self.ny)

        # Set x and y location for all nodes
        for i in range(self.ny):
            self.x[i, :] = np.linspace(Wx[i], Ex[i], self.nx)
            self.y[i, :] = np.linspace(Wy[i], Ey[i], self.nx)

        # Kernel of centroids
        xc, yc = self.get_centroid(self.x, self.y)

        # Kernel of centroids x-coordinates
        self.xc[1:-1, 1:-1] = xc

        # Kernel of centroids y-coordinates
        self.yc[1:-1, 1:-1] = yc

    def compute_normal(self) -> None:

        self.normx = np.zeros((self.ny+1, 2))
        self.normy = np.zeros((self.nx+1, 2))
        print('AAAAAAAAAAAAA')

        theta = np.arctan((self.yc[:, 0] - self.yc[:, -1])/(self.xc[:, -1] - self.xc[:, 0]))
        self.normx[:, 0], self.normx[:, 1] = np.sin(theta), np.cos(theta)

        theta = np.arctan((self.xc[0, :] - self.xc[-1, :])/(self.yc[-1, :] - self.yc[0, :]))
        self.normy[:, 0], self.normy[:, 1] = np.cos(theta), np.sin(theta)

        print(self.normx)
        print(self.normy)

    def compute_cell_area(self):
        """
        Calculates area of every cell in this Block's mesh. A cell is represented as follows:

                              ^
                              | n
                              |
                x--------------------------x
                |             s2        a2 |
                |                          |
        n <-----| s1          O         s3 |-----> n
                |                          |
                | a1          s4           |
                x--------------------------x
                              |
                              | n
                              v

        Each side is labelled s1, s2, s3, s4. a1 and a2 are opposite angles. Note that the sides do not have to be
        alligned with the cartesian axes.

        """

        # Side lengths
        s1 = np.sqrt((self.xc[1:, :-1] - self.xc[:-1, :-1]) ** 2 +
                     (self.yc[1:, :-1] - self.yc[:-1, :-1]) ** 2)

        s3 = np.sqrt((self.xc[1:, 1:] - self.xc[:-1, 1:]) ** 2 +
                     (self.yc[1:, 1:] - self.yc[:-1, 1:]) ** 2)

        s2 = np.sqrt((self.xc[1:, :-1] - self.xc[1:, 1:]) ** 2 +
                     (self.yc[1:, :-1] - self.yc[1:, 1:]) ** 2)

        s4 = np.sqrt((self.xc[:-1, :-1] - self.xc[:-1, 1:]) ** 2 +
                     (self.yc[:-1, :-1] - self.yc[:-1, 1:]) ** 2)

        d1 = (self.xc[1:, :-1] - self.xc[:-1, 1:]) ** 2 + \
             (self.yc[1:, :-1] - self.yc[:-1, 1:]) ** 2

        # Calculate opposite angles
        a1 = np.arccos((s1 ** 2 + s4 ** 2 - d1) / 2 / s1 / s4)
        a2 = np.arccos((s2 ** 2 + s3 ** 2 - d1) / 2 / s2 / s3)

        # Semiperimiter
        s = 0.5 * (s1 + s2 + s3 + s4)

        # Bretschneider formula for quarilateral area
        p1 = (s - s1) * (s - s2) * (s - s3) * (s - s4)
        p2 = s1 * s2 * s3 * s4
        self.A = np.sqrt(p1 - 0.5 * p2 * (1 + np.cos(a1 + a2)))

    @staticmethod
    def get_centroid(x: np.ndarray, y: np.ndarray):

        # Kernel of centroids x-coordinates
        xc = 0.25 * (x[1:, 0:-1] + x[1:, 1:] + x[0:-1, 0:-1] + x[0:-1, 1:])

        # Kernel of centroids y-coordinates
        yc = 0.25 * (y[1:, 0:-1] + y[1:, 1:] + y[0:-1, 0:-1] + y[0:-1, 1:])

        return xc, yc


class Vertices:
    def __init__(self, NE: tuple[Union[float, int], Union[float, int]],
                       NW: tuple[Union[float, int], Union[float, int]],
                       SE: tuple[Union[float, int], Union[float, int]],
                       SW: tuple[Union[float, int], Union[float, int]]) -> None:
        self.NW = NW
        self.NE = NE
        self.SW = SW
        self.SE = SE
