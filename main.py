from pyHype import block
import inputs
import numpy as np
from pyHype import euler_2D

solver = euler_2D.Euler2DExplicitSolver(inputs.E4())
