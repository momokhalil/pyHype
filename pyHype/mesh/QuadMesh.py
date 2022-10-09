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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"

import numpy as np
import matplotlib.pyplot as plt
from pyHype.utils.utils import DirectionalContainerBase
from matplotlib.collections import LineCollection
from pyHype.mesh.base import (
    _mesh_transfinite_gen,
    BlockDescription,
    Vertices,
    CellFace,
    GridLocation,
)


class QuadMesh(_mesh_transfinite_gen):
    def __init__(
        self,
        inputs,
        block_data: BlockDescription = None,
        NE: [float] = None,
        NW: [float] = None,
        SE: [float] = None,
        SW: [float] = None,
        nx: int = None,
        ny: int = None,
    ) -> None:

        if isinstance(block_data, BlockDescription) and (
            NE or NW or SE or SW or nx or ny
        ):
            raise ValueError(
                "Cannot provide block_data of type BlockDescription and also vertices and/or cell count."
            )
        if not isinstance(block_data, BlockDescription) and not (
            NE and NW and SE and SW and nx and ny
        ):
            raise ValueError(
                "If block_data of type BlockDescription is not provided, then vertices for each corner"
                " and cell count must be provided."
            )

        super().__init__()

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
            self.vertices = Vertices(
                NW=block_data.NW, NE=block_data.NE, SW=block_data.SW, SE=block_data.SE
            )
            # Number of cells in x and y directions
            self.nx = inputs.nx
            self.ny = inputs.ny

        # x and y locations of each cell centroid
        self.x = np.zeros((self.ny, self.nx), dtype=float)
        self.y = np.zeros((self.ny, self.nx), dtype=float)
        # Initialize nodes attribute
        self.nodes = None
        # Initialize cell face attributes
        self.face = DirectionalContainerBase()
        # Initialize x and y direction cell sizes
        self.dx = None
        self.dy = None
        # Initialize cell area attribute
        self.A = None
        # Build mesh
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

        # Create cell face classes

        # East Face
        self.face.E = CellFace()
        self.face.E.L = self.east_face_length()
        self.compute_east_face_midpoint()
        self.compute_east_face_norm()
        self.compute_east_face_angle()

        # West Face
        self.face.W = CellFace()
        self.face.W.L = self.west_face_length()
        self.compute_west_face_midpoint()
        self.compute_west_face_norm()
        self.compute_west_face_angle()

        # North Face
        self.face.N = CellFace()
        self.face.N.L = self.north_face_length()
        self.compute_north_face_midpoint()
        self.compute_north_face_norm()
        self.compute_north_face_angle()

        # South Face
        self.face.S = CellFace()
        self.face.S.L = self.south_face_length()
        self.compute_south_face_midpoint()
        self.compute_south_face_norm()
        self.compute_south_face_angle()

        # Cell area
        self.compute_cell_area()

        # Compute dx and dy
        self.dx = self.face.E.xmid - self.face.W.xmid
        self.dy = self.face.N.ymid - self.face.S.ymid

    def compute_east_face_angle(self) -> None:
        """
        Compute and set the east face angles.

        :rtype: None
        :return: None
        """
        self.face.E.theta = self.get_east_face_angle()

    def compute_west_face_angle(self) -> None:
        """
        Compute and set the west face angles.

        :rtype: None
        :return: None
        """
        self.face.W.theta = self.get_west_face_angle()

    def compute_north_face_angle(self) -> None:
        """
        Compute and set the north face angles.

        :rtype: None
        :return: None
        """
        self.face.N.theta = self.get_north_face_angle()

    def compute_south_face_angle(self) -> None:
        """
        Compute and set the south face angles.

        :rtype: None
        :return: None
        """
        self.face.S.theta = self.get_south_face_angle()

    def get_east_face_angle(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the east faces.

        :rtype: np.ndarray
        :return: east face angle
        """
        den = self.nodes.y[1:, 1:] - self.nodes.y[:-1, 1:]
        num = self.nodes.x[:-1, 1:] - self.nodes.x[1:, 1:]
        theta = np.where(den != 0, np.arctan(num / den), 0)
        return theta

    def get_west_face_angle(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the west faces.

        :rtype: np.ndarray
        :return: west face angle
        """
        den = self.nodes.y[1:, :-1] - self.nodes.y[:-1, :-1]
        num = self.nodes.x[:-1, :-1] - self.nodes.x[1:, :-1]
        theta = np.where(den != 0, np.arctan(num / den), 0)
        return theta

    def get_north_face_angle(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the north faces.

        :rtype: np.ndarray
        :return: north face angle
        """
        den = self.nodes.x[1:, 1:] - self.nodes.x[1:, :-1]
        num = self.nodes.y[1:, :-1] - self.nodes.y[1:, 1:]
        theta = np.where(den != 0, np.pi / 2 - np.arctan(num / den), np.pi / 2)
        return theta

    def get_south_face_angle(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the south faces.

        :rtype: np.ndarray
        :return: south face angle
        """
        den = self.nodes.x[:-1, 1:] - self.nodes.x[:-1, :-1]
        num = self.nodes.y[:-1, :-1] - self.nodes.y[:-1, 1:]
        theta = np.where(den != 0, np.pi / 2 - np.arctan(num / den), np.pi / 2)
        return theta

    def get_east_face_norm(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the east faces.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y component of the cell-face outward-facing normal vector.
        """
        theta = self.get_east_face_angle()
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)
        return xnorm, ynorm

    def compute_east_face_norm(self) -> None:
        """
        Sets the normal vector components of the east faces.

        :rtype: None
        :return: None
        """
        self.face.E.xnorm, self.face.E.ynorm = self.get_east_face_norm()

    def get_west_face_norm(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the west faces.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y component of the cell-face outward-facing normal vector.
        """
        theta = self.get_west_face_angle() + np.pi
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)
        return xnorm, ynorm

    def compute_west_face_norm(self) -> None:
        """
        Sets the normal vector components of the west faces.

        :rtype: None
        :return: None
        """
        self.face.W.xnorm, self.face.W.ynorm = self.get_west_face_norm()

    def get_north_face_norm(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the north faces.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y component of the cell-face outward-facing normal vector.
        """
        theta = self.get_north_face_angle()
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)
        return xnorm, ynorm

    def compute_north_face_norm(self) -> None:
        """
        Sets the normal vector components of the north faces.

        :rtype: None
        :return: None
        """
        self.face.N.xnorm, self.face.N.ynorm = self.get_north_face_norm()

    def get_south_face_norm(self) -> [np.ndarray]:
        """
        Calculates and returns the normal vector components of the south faces.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y component of the cell-face outward-facing normal vector.
        """
        theta = self.get_south_face_angle() + np.pi
        xnorm = np.cos(theta)
        ynorm = np.sin(theta)
        return xnorm, ynorm

    def compute_south_face_norm(self) -> None:
        """
        Sets the normal vector components of the south faces.

        :rtype: None
        :return: None
        """
        self.face.S.xnorm, self.face.S.ynorm = self.get_south_face_norm()

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

    @staticmethod
    def get_centroid_from_arrays(x: np.ndarray, y: np.ndarray) -> [np.ndarray]:
        """
        Calculates the centroid coordinates from the x and y nodal coordinates. This is a static method that works on
        any given x and y coordinate arrays.

        :type x: np.ndarray
        :param x: Array of nodal x-coordinates

        :type y: np.array
        :param y: Array of nodal y-coordinates

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: Centroid coordinates from the x and y nodal coordinates
        """
        # Kernel of centroids x-coordinates
        xc = 0.25 * (x[1:, 0:-1] + x[1:, 1:] + x[0:-1, 0:-1] + x[0:-1, 1:])
        # Kernel of centroids y-coordinates
        yc = 0.25 * (y[1:, 0:-1] + y[1:, 1:] + y[0:-1, 0:-1] + y[0:-1, 1:])
        return xc, yc

    def get_centroid(self) -> [np.ndarray]:
        """
        Calculates the centroid coordinates from the x and y nodal coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: Centroid coordinates from the x and y nodal coordinates
        """
        # Kernel of centroids x-coordinates
        x = 0.25 * (
            self.nodes.x[1:, 0:-1]
            + self.nodes.x[1:, 1:]
            + self.nodes.x[0:-1, 0:-1]
            + self.nodes.x[0:-1, 1:]
        )
        # Kernel of centroids x-coordinates
        y = 0.25 * (
            self.nodes.y[1:, 0:-1]
            + self.nodes.y[1:, 1:]
            + self.nodes.y[0:-1, 0:-1]
            + self.nodes.y[0:-1, 1:]
        )
        return x, y

    def compute_centroid(self) -> None:
        """
        Sets the centroid coordinates from the x and y nodal coordinates.

        :rtype: None
        :return: None
        """
        if isinstance(self.nodes, GridLocation):
            self.x, self.y = self.get_centroid()
        else:
            raise AttributeError(
                "Attribute nodes of class Mesh is not of type GridLocation."
            )

    def east_face_length(self) -> np.ndarray:
        """
        Calculates and returns the east face length.

        :rtype: np.ndarray
        :return: east face lengths
        """
        return np.sqrt(
            (
                (self.nodes.x[1:, 1:] - self.nodes.x[:-1, 1:]) ** 2
                + (self.nodes.y[1:, 1:] - self.nodes.y[:-1, 1:]) ** 2
            )
        )

    def west_face_length(self) -> np.ndarray:
        """
        Calculates and returns the west face length.

        :rtype: np.ndarray
        :return: west face lengths
        """
        return np.sqrt(
            (
                (self.nodes.x[1:, :-1] - self.nodes.x[:-1, :-1]) ** 2
                + (self.nodes.y[1:, :-1] - self.nodes.y[:-1, :-1]) ** 2
            )
        )

    def north_face_length(self) -> np.ndarray:
        """
        Calculates and returns the north face length.

        :rtype: np.ndarray
        :return: south north lengths
        """
        return np.sqrt(
            (
                (self.nodes.x[1:, :-1] - self.nodes.x[1:, 1:]) ** 2
                + (self.nodes.y[1:, :-1] - self.nodes.y[1:, 1:]) ** 2
            )
        )

    def south_face_length(self) -> np.ndarray:
        """
        Calculates and returns the south face length.

        :rtype: np.ndarray
        :return: south face lengths
        """
        return np.sqrt(
            (
                (self.nodes.x[:-1, :-1] - self.nodes.x[:-1, 1:]) ** 2
                + (self.nodes.y[:-1, :-1] - self.nodes.y[:-1, 1:]) ** 2
            )
        )

    def get_east_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the east face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the east face midpoint
        """
        x = 0.5 * (self.nodes.x[1:, 1:] + self.nodes.x[:-1, 1:])
        y = 0.5 * (self.nodes.y[1:, 1:] + self.nodes.y[:-1, 1:])
        return x, y

    def get_west_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the west face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the west face midpoint
        """
        x = 0.5 * (self.nodes.x[1:, :-1] + self.nodes.x[:-1, :-1])
        y = 0.5 * (self.nodes.y[1:, :-1] + self.nodes.y[:-1, :-1])
        return x, y

    def get_north_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the north face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the north face midpoint
        """
        x = 0.5 * (self.nodes.x[1:, 1:] + self.nodes.x[1:, :-1])
        y = 0.5 * (self.nodes.y[1:, 1:] + self.nodes.y[1:, :-1])
        return x, y

    def get_south_face_midpoint(self) -> [np.ndarray]:
        """
        Calculates and returns the south face midpoint coordinates.

        :rtype: tuple(np.ndarray, np.ndarray)
        :return: x and y coordinates of the south face midpoint
        """
        x = 0.5 * (self.nodes.x[:-1, 1:] + self.nodes.x[:-1, :-1])
        y = 0.5 * (self.nodes.y[:-1, 1:] + self.nodes.y[:-1, :-1])
        return x, y

    def compute_east_face_midpoint(self) -> None:
        """
        Calculates and sets the east face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.E.xmid, self.face.E.ymid = self.get_east_face_midpoint()

    def compute_west_face_midpoint(self) -> None:
        """
        Calculates and sets the west face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.W.xmid, self.face.W.ymid = self.get_west_face_midpoint()

    def compute_north_face_midpoint(self) -> None:
        """
        Calculates and sets the north face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.N.xmid, self.face.N.ymid = self.get_north_face_midpoint()

    def compute_south_face_midpoint(self) -> None:
        """
        Calculates and sets the south face midpoint coordinates.

        :rtype: None
        :return: None
        """
        self.face.S.xmid, self.face.S.ymid = self.get_south_face_midpoint()

    def east_boundary_angle(self) -> np.ndarray:
        """
        Returns the east boundary angles in radians.

        :rtype: np.ndarray
        :return: East face angles
        """
        return self.face.E.theta[:, -1, None, :]

    def west_boundary_angle(self) -> np.ndarray:
        """
        Returns the west boundary angles in radians.

        :rtype: np.ndarray
        :return: West face angles
        """
        return self.face.W.theta[:, 0, None, :]

    def north_boundary_angle(self) -> np.ndarray:
        """
        Returns the north boundary angles in radians.

        :rtype: np.ndarray
        :return: North face angles
        """
        return self.face.N.theta[-1, None, :, :]

    def south_boundary_angle(self) -> np.ndarray:
        """
        Returns the south boundary angles in radians.

        :rtype: np.ndarray
        :return: South face angles
        """
        return self.face.S.theta[0, None, :, :]

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
