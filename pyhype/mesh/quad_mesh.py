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

from typing import TYPE_CHECKING

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pyhype.utils.utils import SidePropertyDict, NumpySlice
from pyhype.mesh.base import (
    _mesh_transfinite_gen,
    CellFace,
    GridLocation,
)
from pyhype.blocks.base import BlockGeometry

if TYPE_CHECKING:
    from pyhype.solvers.base import SolverConfig


class QuadMesh(_mesh_transfinite_gen):
    def __init__(
        self,
        config: SolverConfig,
        block_geometry: BlockGeometry = None,
    ) -> None:

        super().__init__()

        self.config = config

        self.nghost = block_geometry.nghost
        self.nx = block_geometry.nx
        self.ny = block_geometry.ny
        self.vertices = block_geometry.vertices

        # x and y locations of each cell centroid
        self.x = np.zeros((self.ny, self.nx), dtype=float)
        self.y = np.zeros((self.ny, self.nx), dtype=float)
        self.shape = (self.ny, self.nx)

        self.boundary_index = NumpySlice.boundary()

        self.nodes = None
        self.face = None
        self.dx = None
        self.dy = None
        self.A = None

        self.create_mesh()

    def create_mesh(self) -> None:
        """
        Calculates the nodal and cell-center coordinates, cell face properties, and cell area.

        :rtype: None
        :return: None
        """
        # East edge x and y node locations
        Ex = np.linspace(self.vertices.SE[0], self.vertices.NE[0], self.ny + 1)
        Ey = np.linspace(self.vertices.SE[1], self.vertices.NE[1], self.ny + 1)

        # West edge x and y node locations
        Wx = np.linspace(self.vertices.SW[0], self.vertices.NW[0], self.ny + 1)
        Wy = np.linspace(self.vertices.SW[1], self.vertices.NW[1], self.ny + 1)

        # Initialize temporary storage arrays for x and y node locations
        x = np.zeros((self.ny + 1, self.nx + 1), dtype=float)
        y = np.zeros((self.ny + 1, self.nx + 1), dtype=float)

        # Set x and y location for all nodes
        for i in range(self.ny + 1):
            x[i, :] = np.linspace(Wx[i], Ex[i], self.nx + 1).reshape(
                -1,
            )
            y[i, :] = np.linspace(Wy[i], Ey[i], self.nx + 1).reshape(
                -1,
            )

        # Create nodal location class
        self.nodes = GridLocation(x[:, :, np.newaxis], y[:, :, np.newaxis])

        # Centroid x and y locations
        self.compute_centroid()

        east_high, east_low = (
            GridLocation(self.nodes.x[1:, 1:], self.nodes.y[1:, 1:]),
            GridLocation(self.nodes.x[:-1, 1:], self.nodes.y[:-1, 1:]),
        )
        west_high, west_low = (
            GridLocation(self.nodes.x[1:, :-1], self.nodes.y[1:, :-1]),
            GridLocation(self.nodes.x[:-1, :-1], self.nodes.y[:-1, :-1]),
        )
        north_high, north_low = (
            GridLocation(self.nodes.x[1:, :-1], self.nodes.y[1:, :-1]),
            GridLocation(self.nodes.x[1:, 1:], self.nodes.y[1:, 1:]),
        )
        south_high, south_low = (
            GridLocation(self.nodes.x[:-1, :-1], self.nodes.y[:-1, :-1]),
            GridLocation(self.nodes.x[:-1, 1:], self.nodes.y[:-1, 1:]),
        )

        # Create cell face classes
        self.face = SidePropertyDict(
            E=CellFace(east_high, east_low, orientation="vertical", direction=1),
            W=CellFace(west_high, west_low, orientation="vertical", direction=-1),
            N=CellFace(north_high, north_low, orientation="horizontal", direction=1),
            S=CellFace(south_high, south_low, orientation="horizontal", direction=-1),
        )

        # Cell area
        self.compute_cell_area()

        # Compute dx and dy
        self.dx = self.face.E.midpoint.x - self.face.W.midpoint.x
        self.dy = self.face.N.midpoint.y - self.face.S.midpoint.y

    def compute_cell_area(self) -> None:
        """
        Calculates area of every cell using the Bretschneider formula. A cell is represented as follows:

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

        :rtype: None
        :return: None
        """

        # Side lengths
        s1 = self.west_face_length()
        s3 = self.east_face_length()
        s2 = self.north_face_length()
        s4 = self.south_face_length()

        # Diagonal squared
        d2 = (self.nodes.x[1:, :-1] - self.nodes.x[:-1, 1:]) ** 2 + (
            self.nodes.y[1:, :-1] - self.nodes.y[:-1, 1:]
        ) ** 2

        # Calculate opposite angles
        a1 = np.arccos((s1**2 + s4**2 - d2) / (2 * s1 * s4))
        a2 = np.arccos((s2**2 + s3**2 - d2) / (2 * s2 * s3))

        # Semiperimiter
        s = 0.5 * (s1 + s2 + s3 + s4)

        # Bretschneider formula for quarilateral area
        p1 = (s - s1) * (s - s2) * (s - s3) * (s - s4)
        p2 = s1 * s2 * s3 * s4

        self.A = np.sqrt(p1 - 0.5 * p2 * (1 + np.cos(a1 + a2)))

    def get_centroid(self) -> [np.ndarray]:
        """
        Calculates the centroid coordinates from the x and y nodal coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: Centroid coordinates from the x and y nodal coordinates
        """
        x_ne, y_ne = self.get_NE_vertices()
        x_nw, y_nw = self.get_NW_vertices()
        x_se, y_se = self.get_SE_vertices()
        x_sw, y_sw = self.get_SW_vertices()
        x = 0.25 * (x_ne + x_nw + x_se + x_sw)
        y = 0.25 * (y_ne + y_nw + y_se + y_sw)
        return x, y

    def compute_centroid(self) -> None:
        """
        Sets the centroid coordinates from the x and y nodal coordinates.

        :rtype: None
        :return: None
        """
        if not isinstance(self.nodes, GridLocation):
            raise TypeError(
                "Attribute nodes of class Mesh is not of type GridLocation."
            )
        self.x, self.y = self.get_centroid()

    def east_face_length(self) -> np.ndarray:
        """
        Calculates and returns the east face length.

        :rtype: np.ndarray
        :return: east face lengths
        """
        x_ne, y_ne = self.get_NE_vertices()
        x_se, y_se = self.get_SE_vertices()
        return self._face_length(x_se, x_ne, y_ne, y_se)

    def west_face_length(self) -> np.ndarray:
        """
        Calculates and returns the west face length.

        :rtype: np.ndarray
        :return: west face lengths
        """
        x_nw, y_nw = self.get_NW_vertices()
        x_sw, y_sw = self.get_SW_vertices()
        return self._face_length(x_sw, x_nw, y_nw, y_sw)

    def north_face_length(self) -> np.ndarray:
        """
        Calculates and returns the north face length.

        :rtype: np.ndarray
        :return: south north lengths
        """
        x_ne, y_ne = self.get_NE_vertices()
        x_nw, y_nw = self.get_NW_vertices()
        return self._face_length(x_nw, x_ne, y_ne, y_nw)

    def south_face_length(self) -> np.ndarray:
        """
        Calculates and returns the south face length.

        :rtype: np.ndarray
        :return: south face lengths
        """
        x_se, y_se = self.get_SE_vertices()
        x_sw, y_sw = self.get_SW_vertices()
        return self._face_length(x_sw, x_se, y_se, y_sw)

    @staticmethod
    @nb.njit(cache=True)
    def _face_length(x1, x2, y1, y2):
        res = np.zeros_like(x1)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                for k in range(x1.shape[2]):
                    dx = x1[i, j, k] - x2[i, j, k]
                    dy = y1[i, j, k] - y2[i, j, k]
                    res[i, j, k] = np.sqrt(dx * dx + dy * dy)
        return res

    def get_east_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the east face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the east face midpoint
        """
        x_ne, y_ne = self.get_NE_vertices()
        x_se, y_se = self.get_SE_vertices()
        return self._face_midpoint(x_se, x_ne, y_ne, y_se)

    def get_west_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the west face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the west face midpoint
        """
        x_nw, y_nw = self.get_NW_vertices()
        x_sw, y_sw = self.get_SW_vertices()
        return self._face_midpoint(x_sw, x_nw, y_nw, y_sw)

    def get_north_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the north face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the north face midpoint
        """
        x_ne, y_ne = self.get_NE_vertices()
        x_nw, y_nw = self.get_NW_vertices()
        return self._face_midpoint(x_nw, x_ne, y_ne, y_nw)

    def get_south_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the south face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the south face midpoint
        """
        x_se, y_se = self.get_SE_vertices()
        x_sw, y_sw = self.get_SW_vertices()
        return self._face_midpoint(x_sw, x_se, y_se, y_sw)

    @staticmethod
    @nb.njit(cache=True)
    def _face_midpoint(x1, x2, y1, y2):
        x = np.zeros_like(x1)
        y = np.zeros_like(x1)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                for k in range(x1.shape[2]):
                    x[i, j, k] = 0.5 * (x1[i, j, k] + x2[i, j, k])
                    y[i, j, k] = 0.5 * (y1[i, j, k] + y2[i, j, k])
        return x, y

    def compute_east_face_midpoint(self) -> None:
        """
        Calculates and sets the east face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.E.midpoint.x, self.face.E.midpoint.y = self.get_east_face_midpoint()

    def compute_west_face_midpoint(self) -> None:
        """
        Calculates and sets the west face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.W.midpoint.x, self.face.W.midpoint.y = self.get_west_face_midpoint()

    def compute_north_face_midpoint(self) -> None:
        """
        Calculates and sets the north face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.N.midpoint.x, self.face.N.midpoint.y = self.get_north_face_midpoint()

    def compute_south_face_midpoint(self) -> None:
        """
        Calculates and sets the south face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.S.midpoint.x, self.face.S.midpoint.y = self.get_south_face_midpoint()

    def boundary_angle(self, direction: int) -> np.ndarray:
        """
        Returns the east boundary angles in radians.

        :rtype: np.ndarray
        :return: East face angles
        """
        return self.face[direction].theta[self.boundary_index[direction]]

    def get_NE_vertices(self):
        """
        Returns the x and y coordinates of the North-East cell vertices

        :rtype: np.ndarray
        :return: x and y coordinates of the North-East cell vertices
        """
        return self.nodes.x[1:, 1:, :], self.nodes.y[1:, 1:, :]

    def get_NW_vertices(self):
        """
        Returns the x and y coordinates of the North-West cell vertices

        :rtype: np.ndarray
        :return: x and y coordinates of the North-West cell vertices
        """
        return self.nodes.x[1:, :-1, :], self.nodes.y[1:, :-1, :]

    def get_SE_vertices(self):
        """
        Returns the x and y coordinates of the South-East cell vertices

        :rtype: np.ndarray
        :return: x and y coordinates of the South-East cell vertices
        """
        return self.nodes.x[:-1, 1:, :], self.nodes.y[:-1, 1:, :]

    def get_SW_vertices(self):
        """
        Returns the x and y coordinates of the South-West cell vertices

        :rtype: np.ndarray
        :return: x and y coordinates of the South-West cell vertices
        """
        return self.nodes.x[:-1, :-1, :], self.nodes.y[:-1, :-1, :]

    def plot(self, ax: plt.axes = None):
        """
        Plots the mesh. This function was implemented for debugging purposes.

        :param ax: plt.axes
        :param ax: matplotlib axes object for plotting on an external figure.

        :rtype: None
        :return: None
        """
        if ax is not None:
            ax.scatter(self.nodes.x, self.nodes.y, color="black", s=15)
            ax.scatter(self.x, self.y, color="mediumslateblue", s=15)

            segs1 = np.stack((self.nodes.x, self.nodes.y), axis=2)

            segs2 = segs1.transpose((1, 0, 2))
            plt.gca().add_collection(
                LineCollection(segs1, colors="black", linewidths=1)
            )
            plt.gca().add_collection(
                LineCollection(segs2, colors="black", linewidths=1)
            )

            segs1 = np.stack((self.x, self.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))

            plt.gca().add_collection(
                LineCollection(
                    segs1,
                    colors="mediumslateblue",
                    linestyles="--",
                    linewidths=1,
                    alpha=0.5,
                )
            )
            plt.gca().add_collection(
                LineCollection(
                    segs2,
                    colors="mediumslateblue",
                    linestyles="--",
                    linewidths=1,
                    alpha=0.5,
                )
            )

        else:
            plt.scatter(self.nodes.x, self.nodes.y, color="black", s=15)
            plt.scatter(self.x, self.y, color="mediumslateblue", s=15)

            segs1 = np.stack((self.nodes.x, self.nodes.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))
            plt.gca().add_collection(
                LineCollection(segs1, colors="black", linewidths=1)
            )
            plt.gca().add_collection(
                LineCollection(segs2, colors="black", linewidths=1)
            )

            segs1 = np.stack((self.x, self.y), axis=2)
            segs2 = segs1.transpose((1, 0, 2))

            plt.gca().add_collection(
                LineCollection(
                    segs1,
                    colors="mediumslateblue",
                    linestyles="--",
                    linewidths=1,
                    alpha=0.5,
                )
            )
            plt.gca().add_collection(
                LineCollection(
                    segs2,
                    colors="mediumslateblue",
                    linestyles="--",
                    linewidths=1,
                    alpha=0.5,
                )
            )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
        plt.close()
