import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from .block import Blocks
import pyHype.input_files.input_file_builder as input_file_builder
import pyHype.mesh.mesh_builder as mesh_builder
from pyHype import execution_prints
import cProfile, pstats


class Euler2DSolver:
    def __init__(self, input_dict):

        mesh_inputs = mesh_builder.build(mesh_name=input_dict['mesh_name'],
                                         nx=input_dict['nx'],
                                         ny=input_dict['ny'])

        self._input = input_file_builder.build(input_dict, mesh_inputs)
        self._blocks = Blocks(self._input)

        self.t = 0
        self.numTimeStep = 0
<<<<<<< HEAD:pyHype/euler_2D.py
        self.CFL = self._input.CFL
        self.t_final = self._input.t_final * self._input.a_inf
        self.profile = None
=======
        self.CFL = self.inputs.CFL
        self.t_final = self.inputs.t_final * self.inputs.a_inf
        self.profile = False
        self.profile_data = None
        self.realplot = None
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py

    @property
    def blocks(self):
        return self._blocks.blocks.values()

    def set_IC(self):

        problem_type = self._input.problem_type
        g = self._input.gamma
        ny = self._input.ny
        nx = self._input.nx

        print('    Initial condition type: ', problem_type)

        if problem_type == 'shockbox':

            # High pressure zone
<<<<<<< HEAD:pyHype/euler_2D.py
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
=======
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
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self._blocks.blocks.values():
<<<<<<< HEAD:pyHype/euler_2D.py

                for j in range(1, ny + 1):
                    for i in range(1, nx + 1):
                        iF = 4 * (i - 1) + 4 * nx * (j - 1)
                        iE = 4 * (i - 0) + 4 * nx * (j - 1)

                        if block.mesh.x[j - 1, i - 1] <= 2 and block.mesh.y[j - 1, i - 1] <= 2:
                            block.state.U[iF:iE] = QR
                        elif block.mesh.x[j - 1, i - 1] > 2 and block.mesh.y[j - 1, i - 1] > 2:
                            block.state.U[iF:iE] = QR
=======
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
                        elif block.mesh.x[i, j] > 5 and block.mesh.y[i, j] > 5:
                            block.state.U[i, j, :] = QR
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py
                        else:
                            block.state.U[i, j, :] = QL
                block.state.non_dim()

        elif problem_type == 'implosion':

            # High pressure zone
<<<<<<< HEAD:pyHype/euler_2D.py
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
=======
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
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py
            eR = pR / (g - 1)

            # Create state vectors
            QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
            QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

            # Fill state vector in each block
            for block in self.blocks:
<<<<<<< HEAD:pyHype/euler_2D.py

                for j in range(1, ny + 1):
                    for i in range(1, nx + 1):

                        iF = 4 * (i - 1) + 4 * nx * (j - 1)
                        iE = 4 * (i - 0) + 4 * nx * (j - 1)

                        if block.mesh.x[j - 1, i - 1] < 3 and block.mesh.y[j - 1, i - 1] < 3:
                            block.state.U[iF:iE] = QR
=======
                for i in range(ny):
                    for j in range(nx):
                        if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                            block.state.U[i, j, :] = QR
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py
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
            dt_ = self.CFL * min((block.mesh.dx / (W.u + a)).min(), (block.mesh.dy / (W.v + a)).min())

            if dt_ < dt: dt = dt_

        return dt


    def solve(self):

        print(execution_prints.pyhype)
        print(execution_prints.began_solving + self._input.problem_type)
        print('Date and time: ', datetime.today())

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Initial Conditions')
        self.set_IC()

        print()
        print('----------------------------------------------------------------------------------------')
        print('Setting Boundary Conditions')
        self.set_BC()

<<<<<<< HEAD:pyHype/euler_2D.py
        plt.ion()

        nx = self._input.nx
        ny = self._input.ny

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()
=======
        if self.inputs.realplot:
            plt.ion()
            self.realplot = plt.axes()
            self.realplot.figure.set_size_inches(8, 8)
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py


<<<<<<< HEAD:pyHype/euler_2D.py
        #profiler = cProfile.Profile()
        #profiler.enable()

        print(self.t_final)
=======
        print('Start simulation')
        while self.t < self.t_final:
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py

        while self.t <= self.t_final:

            #print('get dt')
            dt = self.dt()
            self.numTimeStep += 1

            #print('update block')
            self._blocks.update(dt)

<<<<<<< HEAD:pyHype/euler_2D.py
            if self.numTimeStep % 1 == 0:

                state = self._blocks.blocks[1].state.U
=======
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
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py

                V = np.zeros((ny, nx))
                x = self._blocks.blocks[1].mesh.x
                y = self._blocks.blocks[1].mesh.y

<<<<<<< HEAD:pyHype/euler_2D.py
                for i in range(1, ny + 1):
                    Q = state[4 * nx * (i - 1):4 * nx * i]
                    V[i - 1, :] = Q[::4].reshape(-1,)

                ax.contourf(x, y, V)
                plt.show()
                plt.pause(0.01)

            self.t += dt
            print(self.t)
=======
            V = np.zeros((self.inputs.ny, self.inputs.nx))

            for i in range(1, self.inputs.ny + 1):
                Q = state[4 * self.inputs.nx * (i - 1):4 * self.inputs.nx * i]
                V[i - 1, :] = Q[::4].reshape(-1, )

            self.realplot.contourf(self._blocks.blocks[1].mesh.x,
                                   self._blocks.blocks[1].mesh.y,
                                   V, 100, cmap='magma')
            plt.show(block=True)
>>>>>>> parent of e659274 (Revert "revert"):pyHype/solver.py

        #profiler.disable()
        #self.profile = pstats.Stats(profiler)
