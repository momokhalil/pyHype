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

from abc import abstractmethod


class TimeIntegrator:
    def __init__(self, inputs, refBLK):
        self.inputs = inputs
        self.refBLK = refBLK

    def __call__(self, dt):
        self.integrate(dt)

    def get_residual(self):

        # Compute fluxes on each cell face
        self.refBLK.get_flux()

        # Integrate fluxes
        fluxE = self.refBLK.Flux_EW[:, 1:, :]  * self.refBLK.mesh.E_face_L
        fluxW = self.refBLK.Flux_EW[:, :-1, :] * self.refBLK.mesh.W_face_L * (-1)
        fluxN = self.refBLK.Flux_NS[1:, :, :]  * self.refBLK.mesh.N_face_L
        fluxS = self.refBLK.Flux_NS[:-1, :, :] * self.refBLK.mesh.S_face_L * (-1)

        return -(fluxE + fluxW + fluxN + fluxS) / self.refBLK.mesh.A

    # Update state and boundary conditions
    def update_state(self, U):
        self.refBLK.state.update(U)
        self.refBLK.set_BC()

    # Abstract methodo to define integration scheme
    @abstractmethod
    def integrate(self, dt):
        pass
