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

import numpy as np
import matplotlib.pyplot as plt

class NACA4:

    a0 = 0.2969
    a1 = -0.126
    a2 = -0.3516
    a3 = 0.2843
    a4 = -0.1036

    def __init__(self,
                 airfoil: str = '2412',
                 npt: int = 100,
                 angle_start: float = 0,
                 angle_end: float = 180,
                 aoa:float = 0,
                 ):

        # Maximum camber
        self.M = int(airfoil[0]) / 100
        # Location of maximum camber thickness
        self.P = int(airfoil[1]) / 10
        # Thickness
        self.T = int(airfoil[2:]) / 100

        # Angles from start to end for cosine spacing the x-coordinates
        self.beta = np.linspace(np.pi * angle_start / 180,
                                np.pi * angle_end / 180,
                                npt)

        # Cosine spaced x coordinates
        self.x_cam = 0.5 * (1 - np.cos(self.beta))

        # Camber
        self.y_cam = np.where(self.x_cam < self.P,
                              self.front_camber(self.x_cam),
                              self.back_camber(self.x_cam))

        # Camber gradient wrt x
        g_cam = np.where(self.x_cam < self.P,
                         self.front_camber_grad(self.x_cam),
                         self.back_camber_grad(self.x_cam))

        # Thickness distribution
        self.yt = self.thickess_dist(self.T, self.x_cam)
        # Camber angle
        self.theta = np.arctan(g_cam)

        # x and y coordinates of upper and lower surfaces
        self.x_upper = self.x_cam - self.yt * np.sin(self.theta)
        self.y_upper = self.y_cam + self.yt * np.cos(self.theta)
        self.x_lower = self.x_cam + self.yt * np.sin(self.theta)
        self.y_lower = self.y_cam - self.yt * np.cos(self.theta)

        if aoa != 0:
            theta = aoa * np.pi / 180
            xu = self.x_upper * np.cos(theta) + self.y_upper * np.sin(theta)
            yu = self.y_upper * np.cos(theta) - self.x_upper * np.sin(theta)
            xl = self.x_lower * np.cos(theta) + self.y_lower * np.sin(theta)
            yl = self.y_lower * np.cos(theta) - self.x_lower * np.sin(theta)

            self.x_upper, self.y_upper = xu, yu
            self.x_lower, self.y_lower = xl, yl

    def front_camber(self,
                     x: np.ndarray
                     ) -> np.ndarray:
        return (self.M / self.P ** 2) * (2 * self.P * x - x ** 2)

    def front_camber_grad(self,
                          x: np.ndarray
                          ) -> np.ndarray:
        return (2 * self.M / self.P ** 2) * (self.P - x)

    def back_camber(self,
                    x: np.ndarray
                    ) -> np.ndarray:
        return (self.M / (1 - self.P) ** 2) * (1 - 2 * self.P + 2 * self.P * x - x ** 2)

    def back_camber_grad(self,
                         x: np.ndarray
                         ) -> np.ndarray:
        return (2 * self.M / (1 - self.P) ** 2) * (self.P - x)

    def thickess_dist(self,
                      T: float,
                      x: np.ndarray
                      ) -> np.ndarray:

        return T * (self.a0 * np.sqrt(x) +
                    self.a1 * x +
                    self.a2 * x ** 2 +
                    self.a3 * x ** 3 +
                    self.a4 * x ** 4) / 0.2

    @staticmethod
    def cluster_right(start, end, n, factor: float = 2.0):
        length = end - start
        _x = np.linspace(0, 1, n)
        _s = np.tanh(factor * (1 - _x)) / np.tanh(factor)
        _s *= length
        _s += start
        return _s

    @staticmethod
    def cluster_left(start, end, n, factor: float = 2.0):
        length = end - start
        _x = np.linspace(0, 1, n)
        _s = 1 - np.tanh(factor * (1 - _x)) / np.tanh(factor)
        _s *= length
        _s += start
        return _s

    def plot(self):
        plt.plot(self.x_upper, self.y_upper, color='black')
        plt.plot(self.x_lower, self.y_lower, color='black')
        plt.plot(self.x_cam, self.y_cam, color='blue')

        plt.show()
        plt.close()
