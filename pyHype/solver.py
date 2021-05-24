import sys
import pstats
import cProfile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pyHype import execution_prints
from pyHype.blocks.base import Blocks
import pyHype.mesh.mesh_inputs as mesh_inputs
import pyHype.input.input_file_builder as input_file_builder

np.set_printoptions(threshold=sys.maxsize)


class Euler2DSolver:
    def __init__(self, input_dict):

        mesh = mesh_inputs.build(mesh_name=input_dict['mesh_name'],
                                 nx=input_dict['nx'],
                                 ny=input_dict['ny'])

        self.inputs = input_file_builder.ProblemInput(input_dict, mesh)

        self._blocks = Blocks(self.inputs)

        self.t = 0
        self.numTimeStep = 0
        self.CFL = self.inputs.CFL
        self.t_final = self.inputs.t_final * self.inputs.a_inf
        self.profile = False
        self.profile_data = None
        self.realplot = None

    @property
    def blocks(self):
        return self._blocks.blocks.values()

    def set_IC(self):

        problem_type = self.inputs.problem_type
        g = self.inputs.gamma
        ny = self.inputs.ny
        nx = self.inputs.nx

        print('    Initial condition type: ', problem_type)

        if problem_type == 'shockbox':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400.0
            uL = 0.0
            vL = 0.0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 0.0
            vR = 0.0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self._blocks.blocks.values():
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
                        elif block.mesh.x[i, j] > 5 and block.mesh.y[i, j] > 5:
                            block.state.U[i, j, :] = QR
                        else:
                            block.state.U[i, j, :] = QL
                block.state.non_dim()

        elif problem_type == 'implosion':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400.0
            uL = 0.0
            vL = 0.0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100.0
            uR = 0.0
            vR = 0.0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self.blocks:
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
                        else:
                            block.state.U[i, j, :] = QL
                block.state.non_dim()

    def set_BC(self):
        self._blocks.set_BC()

    def dt(self):
        dt = 1000000
        for block in self.blocks:
            W = block.state.to_primitive_state()
            a = W.a()

            t1 = block.mesh.dx / (np.absolute(W.u) + a)
            t2 = block.mesh.dx / (np.absolute(W.v) + a)

            dt_ = self.CFL * min(t1.min(), t2.min())

            if dt_ < dt: dt = dt_

        return dt


    def solve(self):

        print(execution_prints.pyhype)
        print(execution_prints.began_solving + self.inputs.problem_type)
        print('Date and time: ', datetime.today())

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Initial Conditions')
        self.set_IC()

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Boundary Conditions')
        self.set_BC()

        if self.inputs.realplot:
            plt.ion()
            self.realplot = plt.axes()
            self.realplot.figure.set_size_inches(8, 8)

        if self.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        print('Start simulation')
        while self.t < self.t_final:

            dt = self.dt()
            self.numTimeStep += 1

            print('update block')
            self._blocks.update(dt)

            if self.inputs.realplot:
                V = np.zeros((self.inputs.ny, self.inputs.nx))
                if self.numTimeStep % 1 == 0:

                    state = self._blocks.blocks[1].state

                    for i in range(1, self.inputs.ny + 1):
                        Q = state.U[4 * self.inputs.nx * (i - 1):4 * self.inputs.nx * i]
                        V[i - 1, :] = Q[::4].reshape(-1,)

                    self.realplot.contourf(self._blocks.blocks[1].mesh.x,
                                           self._blocks.blocks[1].mesh.y,
                                           V, 20, cmap='magma')
                    plt.show()
                    plt.pause(0.001)

            self.t += dt

        if self.inputs.makeplot:
            state = self._blocks.blocks[1].state.U

            V = np.zeros((self.inputs.ny, self.inputs.nx))

            for i in range(1, self.inputs.ny + 1):
                Q = state[4 * self.inputs.nx * (i - 1):4 * self.inputs.nx * i]
                V[i - 1, :] = Q[::4].reshape(-1, )

            self.realplot.contourf(self._blocks.blocks[1].mesh.x,
                                   self._blocks.blocks[1].mesh.y,
                                   V, 100, cmap='magma')
            plt.show(block=True)

        if self.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
