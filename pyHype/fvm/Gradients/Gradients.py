import numpy as np
from pyHype.states import ConservativeState, PrimitiveState
from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm.Gradients.least_squares import least_squares_9_point


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
