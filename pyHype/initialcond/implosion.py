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

def implosion(self, refBLK):

    # High pressure zone
    rhoL = 4.6968
    pL = 404400.0
    uL = 0.00
    vL = 0.0
    eL = pL / (self.g - 1)

    # Low pressure zone
    rhoR = 1.1742
    pR = 101100.0
    uR = 0.00
    vR = 0.0
    eR = pR / (self.g - 1)

    # Create state vectors
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

    # Fill state vector in each block
    for i in range(refBLK.mesh.ny):
        for j in range(refBLK.mesh.nx):
            if refBLK.mesh.x[i, j] <= 3 and refBLK.mesh.y[i, j] <= 3:
                refBLK.state.U[i, j, :] = QL
            else:
                refBLK.state.U[i, j, :] = QR

    refBLK.state.non_dim()
