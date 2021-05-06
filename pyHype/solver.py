import pstats
import cProfile
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pyHype import execution_prints
from pyHype.blocks.base import Blocks
import pyHype.mesh.mesh_builder as mesh_builder
import pyHype.input.input_file_builder as inputsfile_builder


class Euler2DSolver:
    def __init__(self, inputsdict):

        mesh_inputs = mesh_builder.build(mesh_name=inputsdict['mesh_name'],
                                         nx=inputsdict['nx'],
                                         ny=inputsdict['ny'])

        self.inputs = inputsfile_builder.build(inputsdict, mesh_inputs)
        self._blocks = Blocks(self.inputs)

        self.t = 0
        self.numTimeStep = 0
        self.CFL = self.inputs.CFL
        self.t_final = self.inputs.t_final * self.inputs.a_inf
        self.profile = False
        self.profile_data = None

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
            pL = 404400
            uL = 0
            vL = 0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100
            uR = 0
            vR = 0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((4, 1))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((4, 1))

            # Fill state vector in each block
            for block in self._blocks.blocks.values():

                for j in range(1, ny + 1):
                    for i in range(1, nx + 1):
                        iF = 4 * (i - 1) + 4 * nx * (j - 1)
                        iE = 4 * (i - 0) + 4 * nx * (j - 1)

                        if block.mesh.x[j - 1, i - 1] <= 5 and block.mesh.y[j - 1, i - 1] <= 5:
                            block.state.U[iF:iE] = QR
                        elif block.mesh.x[j - 1, i - 1] > 5 and block.mesh.y[j - 1, i - 1] > 5:
                            block.state.U[iF:iE] = QR
                        else:
                            block.state.U[iF:iE] = QL

                block.state.non_dim()

        elif problem_type == 'implosion':

            # High pressure zone
            rhoL = 4.6968
            pL = 404400
            uL = 0
            vL = 0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 1.1742
            pR = 101100
            uR = 0
            vR = 0
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((4, 1))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((4, 1))

            # Fill state vector in each block
            for block in self.blocks:

                for j in range(1, ny + 1):
                    for i in range(1, nx + 1):

                        iF = 4 * (i - 1) + 4 * nx * (j - 1)
                        iE = 4 * (i - 0) + 4 * nx * (j - 1)

                        if block.mesh.x[j - 1, i - 1] < 5 and block.mesh.y[j - 1, i - 1] < 5:
                            block.state.U[iF:iE] = QR
                        else:
                            block.state.U[iF:iE] = QL

                block.state.non_dim()

    def set_BC(self):
        self._blocks.set_BC()

    def dt(self):
        dt = 1000000
        for block in self.blocks:
            W = block.state.to_W()
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

        plt.ion()

        nx = self.inputs.nx
        ny = self.inputs.ny

        #fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()

        if self.profile:
            print('Enable profiler')
            profiler = cProfile.Profile()
            profiler.enable()
        else:
            profiler = None

        if self.inputs.realplot:
            V = np.zeros((ny, nx))
            x = self._blocks.blocks[1].mesh.x
            y = self._blocks.blocks[1].mesh.y

        print('Start simulation')
        while self.t <= self.t_final:

            #print('get dt')
            dt = self.dt()
            self.numTimeStep += 1

            #print('update block')
            self._blocks.update(dt)

            state = self._blocks.blocks[1].state
            print(np.mean(state.rhou/state.rho))

            if self.inputs.realplot:
                if self.numTimeStep % 1 == 0:

                    state = self._blocks.blocks[1].state.U

                    for i in range(1, ny + 1):
                        Q = state[4 * nx * (i - 1):4 * nx * i]
                        V[i - 1, :] = Q[::4].reshape(-1,)

                    ax.contourf(x, y, V, cmap='magma')
                    plt.show()
                    plt.pause(0.01)

            self.t += dt

        print(self.numTimeStep)

        if self.profile:
            profiler.disable()
            self.profile_data = pstats.Stats(profiler)
