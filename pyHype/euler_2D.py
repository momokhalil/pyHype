import numpy as np
import matplotlib.pyplot as plt
from block import Blocks

class Euler2DExplicitSolver:
    def __init__(self, input_):
        self._input = input_

        self.numTimeStep = 0
        self.t = 0

        self._blocks = Blocks(input_)
