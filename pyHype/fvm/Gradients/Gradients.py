import numpy as np
from pyHype.states import ConservativeState, PrimitiveState
from pyHype.input.input_file_builder import ProblemInput
from pyHype.fvm.Gradients.least_squares import least_squares_nearest_neighbor

class GradientSolver:
    def __init__(self, inputs: ProblemInput):
        self.inputs = inputs


    def least_squares_nearest_neighbor(self, refBLK):

        bdr = refBLK.boundary_blocks

        dQdx, dQdy = least_squares_nearest_neighbor(refBLK.state.Q,
                                                    bdr.E.state.Q, bdr.W.state.Q, bdr.N.state.Q, bdr.S.state.Q,
                                                    refBLK.mesh.x, refBLK.mesh.y,
                                                    bdr.E.x, bdr.E.y, bdr.W.x, bdr.W.y,
                                                    bdr.N.x, bdr.N.y, bdr.S.x, bdr.S.y,
                                                    self.inputs.nx, self.inputs.ny)
        return dQdx, dQdy


