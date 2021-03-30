import numpy as np
import matplotlib.pyplot as plt
from .block import Blocks
import pyHype.input_files.input_file_builder as input_file_builder
import pyHype.mesh.mesh_builder as mesh_builder

class Euler2DSolver:
    def __init__(self, input_dict):
        print(input_dict['mesh_name'] == 'one_mesh')
        mesh_inputs = mesh_builder.build(mesh_name=input_dict['mesh_name'],
                                         nx=input_dict['nx'],
                                         ny=input_dict['ny'])

        self._input = input_file_builder.build(input_dict, mesh_inputs)
        self._blocks = Blocks(self._input)
        self.numTimeStep = 0
        self.t = 0
        self.t_final = self._input.t_final * self._input.a_inf
        self.CFL = self._input.CFL

    @property
    def blocks(self):
        return self._blocks.blocks.values()

    def set_IC(self):

        problem_type = self._input.problem_type
        g = self._input.gamma
        ny = self._input.ny
        nx = self._input.nx

        if problem_type == 'shockbox':

            # High pressure zone
            rhoL = 0.16214
            pL = 404400
            uL = 0
            vL = 0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 0.04053
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

                        if block.mesh.x[j - 1, i - 1] <= 2 and block.mesh.y[j - 1, i - 1] <= 2:
                            block.state.U[iF:iE] = QR
                        elif block.mesh.x[j - 1, i - 1] > 2 and block.mesh.y[j - 1, i - 1] > 2:
                            block.state.U[iF:iE] = QR
                        else:
                            block.state.U[iF:iE] = QL

                block.state.non_dim()

        elif problem_type == 'implosion':

            # High pressure zone
            rhoL = 0.16214
            pL = 404400
            uL = 0
            vL = 0
            eL = pL / (g - 1)

            # Low pressure zone
            rhoR = 0.04053
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
        dt = []
        for block in self.blocks:
            W = block.state.to_W()
            a = W.a()
            dt.append(self.CFL * min((block.mesh.dx / (W.u + a)).min(), (block.mesh.dy / (W.v + a)).min()))
        return min(dt)


    def solve(self):

        print('IN SOLVE')

        print('Set IC')
        self.set_IC()

        print('SET BC')
        self.set_BC()

        plt.ion()

        print('nx')
        nx = self._input.nx
        print('ny')
        ny = self._input.ny

        print('fig')
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()

        while self.t <= self.t_final:

            print('get dt')
            dt = self.dt()
            self.numTimeStep += 1

            print('update block')
            self._blocks.update(dt)

            if self.numTimeStep % 1 == 0:

                state = self._blocks.blocks[1].state.U

                V = np.zeros((ny, nx))
                x = self._blocks.blocks[1].mesh.x
                y = self._blocks.blocks[1].mesh.y

                for i in range(1, ny + 1):
                    Q = state[4 * nx * (i - 1):4 * nx * i]
                    V[i - 1, :] = Q[::4].reshape(-1,)

                ax.contourf(x, y, V)
                plt.show()
                plt.pause(0.01)

            self.t += dt
