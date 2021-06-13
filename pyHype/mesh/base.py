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


class Vertices:
    def __init__(self, NE: tuple[Union[float, int], Union[float, int]],
                       NW: tuple[Union[float, int], Union[float, int]],
                       SE: tuple[Union[float, int], Union[float, int]],
                       SW: tuple[Union[float, int], Union[float, int]]) -> None:
        self.NW = NW
        self.NE = NE
        self.SW = SW
        self.SE = SE


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

        self.EW_norm_x = np.zeros((1, self.nx + 1))
        self.EW_norm_y = np.zeros((1, self.nx + 1))

        self.NS_norm_x = np.zeros((self.ny + 1, 1))
        self.NS_norm_y = np.zeros((self.ny + 1, 1))

        self.thetax = np.zeros((self.nx + 1))
        self.thetay = np.zeros((self.ny + 1))

        self.EW_midpoint_x = None
        self.EW_midpoint_y = None
        self.NS_midpoint_x = None
        self.NS_midpoint_y = None

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
        """
        Computes the x and y components of the normal vector for each cell boundary. For storage efficiency, the results
        are stored for each mesh line in the x and y directions. To make this clear, consider the following 3 x 3 mesh:
                      ^
                      |nx3
            3   O---------O---------O
                |ny1      |ny2      |ny3
                |-->  ^   |-->      |-->
                |     |nx2|         |
            2   O---------O---------O
                |         |         |
                |     ^   |         |
                |     |nx1|         |
            1   O---------O---------O

                1         2         3

        where nx1 is the normal vector for the horizontal line of nodes 1, nx2 is the normal vector for the horizontal
        line of nodes 2, etc. Same applies for ny1, ny2...etc.

        Note that the grid does not have to be alligned with the cartesian axis, this example was done so for clarity.

        The resulting normx and normy vectors are as such:

        normx = [nx1_x, nx1_y
                 nx2_x, nx2_y
                 nx3_x, nx3_y]

        normy = [ny1_x, ny1_y
                 ny2_x, ny2_y
                 ny3_x, ny3_y]

        """

        # Angle and normal vector for y-aligned nodes (used for left and right sides of cells)

        # Numerator and denominator for arctan
        num = self.xc[1, :] - self.xc[-2, :]
        den = self.yc[-2, :] - self.yc[1, :]

        # Check for zero denominator
        _non_zero_den = den != 0
        _zero_den = den == 0

        # Calculate thetax
        self.thetax[_non_zero_den] = np.arctan(num[_non_zero_den] / den[_non_zero_den])
        self.thetax[_zero_den] = 0

        # Calculate normal vector
        self.EW_norm_x[0, :] = np.cos(self.thetax)
        self.EW_norm_y[0, :] = np.sin(self.thetax)

        # Angle and normal vector for x-aligned nodes (used for top and bottom sides of cells)

        # Numerator and denominator for arctan
        num = self.yc[:, 1] - self.yc[:, -2]
        den = self.xc[:, -2] - self.xc[:, 1]

        # Check for zero denominator
        _non_zero_den = den != 0
        _zero_den = den == 0

        # Calculate thetax
        self.thetay[_non_zero_den] = np.pi / 2 - np.arctan(num[_non_zero_den] / den[_non_zero_den])
        self.thetay[_zero_den] = np.pi / 2

        # Calculate normal vector
        self.NS_norm_x[:, 0] = np.cos(self.thetay)
        self.NS_norm_y[:, 0] = np.sin(self.thetay)


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
        s1 = self.west_side_length()
        s3 = self.east_side_length()
        s2 = self.north_side_length()
        s4 = self.south_side_length()

        # Diagonal
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

    def east_side_length(self):
        """print('-----------------------------------------------')
        print(self.x)
        print(self.y)
        print('-----------------------------------------------')
        print(self.xc)
        print(self.yc)
        print(self.yc[1:, 1:])
        print(self.yc[:-1, 1:])"""
        return np.sqrt(((self.xc[1:, 1:] - self.xc[:-1, 1:]) ** 2 +
                        (self.yc[1:, 1:] - self.yc[:-1, 1:]) ** 2))

    def west_side_length(self):
        return np.sqrt(((self.xc[1:, :-1] - self.xc[:-1, :-1]) ** 2 +
                        (self.yc[1:, :-1] - self.yc[:-1, :-1]) ** 2))

    def north_side_length(self):
        return np.sqrt(((self.xc[1:, :-1] - self.xc[1:, 1:]) ** 2 +
                        (self.yc[1:, :-1] - self.yc[1:, 1:]) ** 2))

    def south_side_length(self):
        return np.sqrt(((self.xc[:-1, :-1] - self.xc[:-1, 1:]) ** 2 +
                        (self.yc[:-1, :-1] - self.yc[:-1, 1:]) ** 2))

    def get_EW_face_midpoint(self):

        x = 0.5 * (self.xc[1:, :] + self.xc[:-1, :])
        y = 0.5 * (self.yc[1:, :] + self.yc[:-1, :])

        return x[:, :, np.newaxis], y[:, :, np.newaxis]

    def get_NS_face_midpoint(self):

        x = 0.5 * (self.xc[:, 1:] + self.xc[:, :-1])
        y = 0.5 * (self.yc[:, 1:] + self.yc[:, :-1])

        return x[:, :, np.newaxis], y[:, :, np.newaxis]
