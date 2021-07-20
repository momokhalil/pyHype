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
from pyHype.states import ConservativeState, PrimitiveState
from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm.Gradients.least_squares import least_squares_9_point


class Gradient:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs


class LeastSquares9Point:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

        self.stencilSW = [[0, 0], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [1, 1]]
        self.stencilNW = [[-2, 0], [-1, 0], [0, 0], [0, 1], [-2, 0], [-2, 1], [-1, 1]]
        self.stencilSE = [[0, 0], [1, 0], [0, -1], [0, -2], [0, -1], [1, -1], [1, -2]]
        self.stencilNE = [[0, -1], [0, -2], [-1, 0], [-2, 0], [-1, -2], [1, -1], [1, -2]]

    def __call__(self, refBLK):
        return self.least_squares_nearest_neighbor(refBLK)

    def least_squares_nearest_neighbor(self, refBLK):
        bdr = refBLK.boundary_blocks

        dQdx, dQdy = least_squares_9_point(refBLK.state.Q,
                                           bdr.E.state.Q, bdr.W.state.Q, bdr.N.state.Q, bdr.S.state.Q,
                                           refBLK.mesh.x, refBLK.mesh.y,
                                           bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                           bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                           self.inputs.nx, self.inputs.ny,
                                           self.stencilSW, self.stencilNW, self.stencilSE, self.stencilNE)
        return dQdx, dQdy


class GreenGauss:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs

    def __call__(self, refBLK):
        return self.green_gauss(refBLK)

    def green_gauss(self, refBLK):

        # Concatenate mesh state and ghost block states
        interfaceEW, interfaceNS = refBLK.get_interface_values()

        # Get each face's contribution to dUdx
        E, W, N, S = self.face_contribution(interfaceEW, interfaceNS, refBLK)

        # Compute dUdx
        refBLK.gradx = (E * refBLK.mesh.faceE.xnorm +
                        W * refBLK.mesh.faceW.xnorm +
                        N * refBLK.mesh.faceN.xnorm +
                        S * refBLK.mesh.faceS.xnorm
                        ) / refBLK.mesh.A

        # Compute dUdy
        refBLK.grady = (E * refBLK.mesh.faceE.ynorm +
                        W * refBLK.mesh.faceW.ynorm +
                        N * refBLK.mesh.faceN.ynorm +
                        S * refBLK.mesh.faceS.ynorm
                        ) / refBLK.mesh.A


    @staticmethod
    def face_contribution(interfaceEW, interfaceNS, refBLK):
        E = interfaceEW[:, 1:, :] * refBLK.mesh.faceE.L
        W = interfaceEW[:, :-1, :] * refBLK.mesh.faceW.L
        N = interfaceNS[1:, :, :] * refBLK.mesh.faceN.L
        S = interfaceNS[:-1, :, :] * refBLK.mesh.faceS.L

        return E, W, N, S
