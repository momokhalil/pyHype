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

import numpy as np
from abc import abstractmethod


class TimeIntegrator:
    def __init__(self, inputs, refBLK):
        self.inputs = inputs
        self.refBLK = refBLK

    def __call__(self, dt):
        self.integrate(dt)

    # Calculate residuals in x and y directions
    def get_residual(self):
        self.refBLK.get_flux()
        return -self.refBLK.Flux_X / self.refBLK.mesh.dx - self.refBLK.Flux_Y / self.refBLK.mesh.dy

    # Update state and boundary conditions
    def update_state(self, U):
        self.refBLK.state.update(U)
        self.refBLK.set_BC()

    # Abstract methodo to define integration scheme
    @abstractmethod
    def integrate(self, dt):
        pass


class ExplicitRungeKutta(TimeIntegrator):
    def __init__(self, inputs, refBLK):

        # Call superclass constructor
        super().__init__(inputs, refBLK)
        # Butcher tableau representation
        self.a = None
        # Number of stages
        self.num_stages = 0

    def integrate(self, dt):

        # Save state vector
        U = self.refBLK.state.U.copy()
        # Initialise dictionary to store stage residuals
        _stage_residuals = {}
        # Iterate across num_stages
        for stage in range(self.num_stages):
            # Get residual for current stage
            _stage_residuals[stage] = self.get_residual()
            # Copy U into intermediate stage
            _intermediate_state = U
            # Add stage contributions to current stage according to the method's Butcher tableau
            for step in range(stage+1):
                # Only add if tableau factor is non-zero
                if self.a[stage][step] != 0:
                    _intermediate_state = _intermediate_state + dt * self.a[stage][step] * _stage_residuals[step]
            # Update state using intermediate state
            self.update_state(_intermediate_state)
