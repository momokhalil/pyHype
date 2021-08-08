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
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
from typing import Union
import matplotlib.pyplot as plt
import pyHype.mesh.airfoil as airfoil
from matplotlib.collections import LineCollection
from matplotlib.pyplot import axes


class BlockDescription:
    def __init__(self, blk_input):

        # Set parameter attributes from input dict
        self.nBLK = blk_input['nBLK']
        self.n = blk_input['n']
        self.nx = blk_input['nx']
        self.ny = blk_input['ny']
        self.nghost = blk_input['nghost']
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


class GridLocation:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray):
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


class Mesh:
    def __init__(self,
                 inputs,
                 block_data: BlockDescription = None,
                 NE: [float] = None,
                 NW: [float] = None,
                 SE: [float] = None,
                 SW: [float] = None,
                 nx: int = None,
                 ny: int = None,
                 ) -> None:

        if isinstance(block_data, BlockDescription) and (NE or NW or SE or SW or nx or ny):
            raise ValueError('Cannot provide block_data of type BlockDescription and also vertices and/or cell count.')

        elif not isinstance(block_data, BlockDescription) and not (NE and NW and SE and SW and nx and ny):
            raise ValueError('If block_data of type BlockDescription is not provided, then vertices for each corner'
                             ' and cell count must be provided.')

        # Initialize inputs class
        self.inputs = inputs

        # Number of ghost cells
        self.nghost = inputs.nghost

        # If block_data is not given
        if not isinstance(block_data, BlockDescription):
            # Initialize vertices class
            self.vertices = Vertices(NW=NW, NE=NE, SW=SW, SE=SE)

            # Number of cells in x and y directions
            self.nx = nx
            self.ny = ny

        else:
            # Initialize vertices class
            self.vertices = Vertices(NW=block_data.NW,
                                     NE=block_data.NE,
                                     SW=block_data.SW,
                                     SE=block_data.SE)

            # Number of cells in x and y directions
            self.nx = inputs.nx
            self.ny = inputs.ny

        # x and y locations of each cell centroid
        self.x = np.zeros((self.ny, self.nx))
        self.y = np.zeros((self.ny, self.nx))

        # Initialize nodes attribute
        self.nodes = None

        # Initialize cell face attributes
        self.faceE = None
        self.faceW = None
        self.faceN = None
        self.faceS = None

        # Initialize x and y direction cell sizes
        self.dx = None
        self.dy = None

        # Initialize cell area attribute
        self.A = None

        # Build mesh
        self.create_mesh()


    def create_mesh(self):

        # --------------------------------------------------------------------------------------------------------------
        # Build node and cell coordinates

        # East edge x and y node locations
        Ex = np.linspace(self.vertices.SE[0], self.vertices.NE[0], self.ny + 1)
        Ey = np.linspace(self.vertices.SE[1], self.vertices.NE[1], self.ny + 1)

        # West edge x and y node locations
        Wx = np.linspace(self.vertices.SW[0], self.vertices.NW[0], self.ny + 1)
        Wy = np.linspace(self.vertices.SW[1], self.vertices.NW[1], self.ny + 1)

        # Initialize temporary storage arrays for x and y node locations
        x = np.zeros((self.ny + 1, self.nx + 1))
        y = np.zeros((self.ny + 1, self.nx + 1))

        # Set x and y location for all nodes
        for i in range(self.ny + 1):
            x[i, :] = np.linspace(Wx[i], Ex[i], self.nx + 1).reshape(-1, )
            y[i, :] = np.linspace(Wy[i], Ey[i], self.nx + 1).reshape(-1, )

        # Create nodal location class
        self.nodes = GridLocation(x[:, :, np.newaxis], y[:, :, np.newaxis])

        # Centroid x and y locations
        self.compute_centroid()

        # --------------------------------------------------------------------------------------------------------------
        # Create cell face classes

        # East Face
        self.faceE = CellFace()
        self.faceE.L = self.east_face_length()
        self.compute_east_face_midpoint()
        self.compute_east_face_norm()

        # West Face
        self.faceW = CellFace()
        self.faceW.L = self.west_face_length()
        self.compute_west_face_midpoint()
        self.compute_west_face_norm()

        # North Face
        self.faceN = CellFace()
        self.faceN.L = self.north_face_length()
        self.compute_north_face_midpoint()
        self.compute_north_face_norm()

        # South Face
        self.faceS = CellFace()
        self.faceS.L = self.south_face_length()
        self.compute_south_face_midpoint()
        self.compute_south_face_norm()

        # Cell area
        self.compute_cell_area()

        # Compute dx and dy
        self.dx = self.faceE.xmid - self.faceW.xmid
        self.dy = self.faceN.ymid - self.faceS.ymid


    def _get_interior_mesh_transfinite(self,
                                       x: np.ndarray,
                                       y: np.ndarray):

        _im = np.linspace(1 / self.ny, (self.ny - 1) / self.ny, self.ny - 2)
        _jm = np.linspace(1 / self.nx, (self.nx - 1) / self.nx, self.nx - 2)

        jm, im = np.meshgrid(_jm, _im)

        _mi = np.linspace((self.ny - 1) / self.ny, 1 / self.ny, self.ny - 2)
        _mj = np.linspace((self.nx - 1) / self.nx, 1 / self.nx, self.nx - 2)

        mj, mi = np.meshgrid(_mj, _mi)

        x[1:-1, 1:-1] = self.__get_kernel_transfinite(x, im, jm, mi, mj)
        x[1:-1, 1:-1] = self.__get_kernel_transfinite(y, im, jm, mi, mj)

        return x, y

    @staticmethod
    def __get_kernel_transfinite(x: np.ndarray,
                                 im: np.ndarray,
                                 jm: np.ndarray,
                                 mi: np.ndarray,
                                 mj: np.ndarray,
                                 ):

        return mi * x[0, 1:-1] + im * x[-1, 1:-1]               \
             + mj * x[1:-1, 0, None] + jm * x[1:-1, -1, None]   \
             - mi * mj * x[0, 0] - mi * jm * x[0, -1]           \
             - im * mj * x[-1, 0]  - im * jm * x[-1, -1]

    def get_east_face_norm(self) -> [np.ndarray]:

        den = self.nodes.y[1:, 1:] - self.nodes.y[:-1, 1:]
        num = self.nodes.x[:-1, 1:] - self.nodes.x[1:, 1:]
        theta = np.where(den != 0, np.arctan(num / den), 0)
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)

        return xnorm, ynorm, theta


    def compute_east_face_norm(self) -> None:

        if isinstance(self.faceE, CellFace):
            self.faceE.xnorm, self.faceE.ynorm, self.faceE.theta = self.get_east_face_norm()
        else:
            raise TypeError('faceE is not of type CellFace')


    def get_west_face_norm(self):

        den = self.nodes.y[1:, :-1] - self.nodes.y[:-1, :-1]
        num = self.nodes.x[:-1, :-1] - self.nodes.x[1:, :-1]
        theta = np.where(den != 0, np.arctan(num / den), 0) + np.pi
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)

        return xnorm, ynorm, theta


    def compute_west_face_norm(self):

        if isinstance(self.faceW, CellFace):
            self.faceW.xnorm, self.faceW.ynorm, self.faceW.theta = self.get_west_face_norm()
        else:
            raise TypeError('faceW is not of type CellFace')


    def get_north_face_norm(self):

        den = self.nodes.x[1:, 1:] - self.nodes.x[1:, :-1]
        num = self.nodes.y[1:, :-1] - self.nodes.y[1:, 1:]
        theta = np.where(den != 0, np.pi / 2 - np.arctan(num / den), np.pi / 2)
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)

        return xnorm, ynorm, theta


    def compute_north_face_norm(self):

        if isinstance(self.faceN, CellFace):
            self.faceN.xnorm, self.faceN.ynorm, self.faceN.theta = self.get_north_face_norm()
        else:
            raise TypeError('faceN is not of type CellFace')


    def get_south_face_norm(self):

        den = self.nodes.x[:-1, 1:] - self.nodes.x[:-1, :-1]
        num = self.nodes.y[:-1, :-1] - self.nodes.y[:-1, 1:]
        theta = np.where(den != 0, np.pi / 2 - np.arctan(num / den), np.pi / 2) + np.pi
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)

        return xnorm, ynorm, theta


    def compute_south_face_norm(self):

        if isinstance(self.faceS, CellFace):
            self.faceS.xnorm, self.faceS.ynorm, self.faceS.theta = self.get_south_face_norm()
        else:
            raise TypeError('faceS is not of type CellFace')


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
        s1 = self.west_face_length()
        s3 = self.east_face_length()
        s2 = self.north_face_length()
        s4 = self.south_face_length()

        # Diagonal
        d1 = (self.nodes.x[1:, :-1] - self.nodes.x[:-1, 1:]) ** 2 + \
             (self.nodes.y[1:, :-1] - self.nodes.y[:-1, 1:]) ** 2

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
    def get_centroid_from_arrays(x: np.ndarray, y: np.ndarray):

        xc = 0.25 * (x[1:, 0:-1] + x[1:, 1:] + x[0:-1, 0:-1] + x[0:-1, 1:])

        # Kernel of centroids y-coordinates
        yc = 0.25 * (y[1:, 0:-1] + y[1:, 1:] + y[0:-1, 0:-1] + y[0:-1, 1:])

        return xc, yc


    def get_centroid(self):
        # Kernel of centroids x-coordinates
        x = 0.25 * (self.nodes.x[1:, 0:-1] +
                    self.nodes.x[1:, 1:] +
                    self.nodes.x[0:-1, 0:-1] +
                    self.nodes.x[0:-1, 1:])

        # Kernel of centroids x-coordinates
        y = 0.25 * (self.nodes.y[1:, 0:-1] +
                    self.nodes.y[1:, 1:] +
                    self.nodes.y[0:-1, 0:-1] +
                    self.nodes.y[0:-1, 1:])

        return x, y


    def compute_centroid(self):

        if isinstance(self.nodes, GridLocation):
            self.x, self.y = self.get_centroid()
        else:
            raise AttributeError('Attribute nodes of class Mesh is not of type GridLocation.')

    # Face Lengths

    def east_face_length(self):
        return np.sqrt(((self.nodes.x[1:, 1:] - self.nodes.x[:-1, 1:]) ** 2 +
                        (self.nodes.y[1:, 1:] - self.nodes.y[:-1, 1:]) ** 2))

    def west_face_length(self):
        return np.sqrt(((self.nodes.x[1:, :-1] - self.nodes.x[:-1, :-1]) ** 2 +
                        (self.nodes.y[1:, :-1] - self.nodes.y[:-1, :-1]) ** 2))

    def north_face_length(self):
        return np.sqrt(((self.nodes.x[1:, :-1] - self.nodes.x[1:, 1:]) ** 2 +
                        (self.nodes.y[1:, :-1] - self.nodes.y[1:, 1:]) ** 2))

    def south_face_length(self):
        return np.sqrt(((self.nodes.x[:-1, :-1] - self.nodes.x[:-1, 1:]) ** 2 +
                        (self.nodes.y[:-1, :-1] - self.nodes.y[:-1, 1:]) ** 2))

    # Face Midpoints

    def get_east_face_midpoint(self):

        x = 0.5 * (self.nodes.x[1:, 1:] + self.nodes.x[:-1, 1:])
        y = 0.5 * (self.nodes.y[1:, 1:] + self.nodes.y[:-1, 1:])

        return x, y

    def get_west_face_midpoint(self):

        x = 0.5 * (self.nodes.x[1:, :-1] + self.nodes.x[:-1, :-1])
        y = 0.5 * (self.nodes.y[1:, :-1] + self.nodes.y[:-1, :-1])

        return x, y

    def get_north_face_midpoint(self):

        x = 0.5 * (self.nodes.x[1:, 1:] + self.nodes.x[1:, :-1])
        y = 0.5 * (self.nodes.y[1:, 1:] + self.nodes.y[1:, :-1])

        return x, y

    def get_south_face_midpoint(self):

        x = 0.5 * (self.nodes.x[:-1, 1:] + self.nodes.x[:-1, :-1])
        y = 0.5 * (self.nodes.y[:-1, 1:] + self.nodes.y[:-1, :-1])

        return x, y

    def compute_east_face_midpoint(self):

        self.faceE.xmid, self.faceE.ymid = self.get_east_face_midpoint()

    def compute_west_face_midpoint(self):

        self.faceW.xmid, self.faceW.ymid = self.get_west_face_midpoint()

    def compute_north_face_midpoint(self):

        self.faceN.xmid, self.faceN.ymid = self.get_north_face_midpoint()

    def compute_south_face_midpoint(self):

        self.faceS.xmid, self.faceS.ymid = self.get_south_face_midpoint()

    # Face Angles

    def get_east_face_angle(self):

        return self.faceE.theta[:, -1, None, :]

    def get_west_face_angle(self):

        return self.faceW.theta[:, 0, None, :] - np.pi

    def get_north_face_angle(self):

        return self.faceN.theta[-1, None, :, :]

    def get_south_face_angle(self):

        return self.faceS.theta[0, None, :, :] - np.pi

    # Plotting

    def plot(self, ax: plt.axes = None):

        if ax is not None:
            ax.scatter(self.nodes.x, self.nodes.y, color='black', s=15)
            ax.scatter(self.x, self.y, color='mediumslateblue', s=15)

            segs1 = np.stack((self.nodes.x, self.nodes.y), axis=2)

            segs2 = segs1.transpose((1, 0, 2))
            plt.gca().add_collection(LineCollection(segs1, colors='black', linewidths=1))
            plt.gca().add_collection(LineCollection(segs2, colors='black', linewidths=1))

            segs1 = np.stack((self.x, self.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))

            plt.gca().add_collection(
                LineCollection(segs1, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))
            plt.gca().add_collection(
                LineCollection(segs2, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))

        else:
            plt.scatter(self.nodes.x, self.nodes.y, color='black', s=15)
            plt.scatter(self.x, self.y, color='mediumslateblue', s=15)

            segs1 = np.stack((self.nodes.x, self.nodes.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))
            plt.gca().add_collection(LineCollection(segs1, colors='black', linewidths=1))
            plt.gca().add_collection(LineCollection(segs2, colors='black', linewidths=1))

            segs1 = np.stack((self.x, self.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))

            plt.gca().add_collection(
                LineCollection(segs1, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))
            plt.gca().add_collection(
                LineCollection(segs2, colors='mediumslateblue', linestyles='--', linewidths=1, alpha=0.5))

        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        plt.close()
