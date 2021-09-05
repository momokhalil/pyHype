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

_DEFINED_IC_ = ['explosion',
                'explosion_3',
                'implosion',
                'shockbox',
                'supersonic_flood',
                'supersonic_rest',
                'subsonic_flood',
                'subsonic_rest',
                'explosion_trapezoid',
                'mach_reflection'
                ]

def is_defined_IC(name: str):
    return True if name in _DEFINED_IC_ else False

def explosion(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

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
    for block in blocks:
        block.state.U[:, :, :] = np.where(np.logical_and(3 <= block.mesh.x <= 7, 3 <= block.mesh.y <= 7), QL, QR)
        block.state.set_vars_from_state()
        block.state.non_dim()


def implosion(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

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
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

    # Fill state vector in each block
    for block in blocks:
        block.state.U[:, :, :] = np.where(np.logical_and(block.mesh.x <= 5, block.mesh.y <= 5), QR, QL)
        block.state.set_vars_from_state()
        block.state.non_dim()


def shockbox(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rhoL = 4.6968
    pL = 404400.0
    uL = 0.00
    vL = 0.0
    eL = pL / (g - 1)

    # Low pressure zone
    rhoR = 1.1742
    pR = 101100.0
    uR = 0.00
    vR = 0.0
    eR = pR / (g - 1)

    # Create state vectors
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.nx):
                if block.mesh.x[i, j] <= 5 and block.mesh.y[i, j] <= 5:
                    block.state.U[i, j, :] = QR
                elif block.mesh.x[i, j] > 5 and block.mesh.y[i, j] > 5:
                    block.state.U[i, j, :] = QR
                else:
                    block.state.U[i, j, :] = QL

        block.state.non_dim()


def mach_reflection(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # Free stream
    rhoL = 8
    pL = 116.5
    uL = 8.25
    vL = 0.0
    eL = pL / (g - 1) + 0.5 * (uL ** 2 + vL ** 2) * rhoL

    # Post shock
    rhoR = 1.4
    pR = 1.0
    uR = 0.0
    vR = 0.0
    eR = pR / (g - 1) + 0.5 * (uR ** 2 + vR ** 2) * rhoR

    # Create state vectors
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL]).reshape((1, 1, 4))
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR]).reshape((1, 1, 4))

    # Fill state vector in each block
    for block in blocks:
        block.state.U[:, :, :] = np.where(block.mesh.x <= 0.95, QL, QR)
        block.state.non_dim()

def supersonic_flood(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rho = 1.0
    p = 1 / g
    u = 2.0
    v = 0.0
    e = p / (g - 1) + 0.5 * (u ** 2 + v ** 2) * rho

    # Create state vectors
    Q = np.array([rho, rho * u, rho * v, e]).reshape((1, 1, 4))

    # Fill state vector in each block
    for block in blocks:
        block.state.U[:, :, :] = Q
        block.state.non_dim()

def subsonic_flood(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rho = 1.0
    p = 1 / g
    u = 0.5
    v = 0.0
    e = p / (g - 1) + 0.5 * (u ** 2 + v ** 2) * rho

    # Create state vectors
    Q = np.array([rho, rho * u, rho * v, e])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.nx):
                block.state.U[i, j, :] = Q

        block.state.non_dim()

def supersonic_rest(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rho = 1.0
    p = 1 / g
    u = 0.0
    v = 0.0
    e = p / (g - 1) + 0.5 * (u ** 2 + v ** 2) * rho

    # Create state vectors
    Q = np.array([rho, rho * u, rho * v, e])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.nx):
                block.state.U[i, j, :] = Q

        block.state.non_dim()

def subsonic_rest(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rho = 1.0
    p = 1 / g
    u = 0.0
    v = 0.0
    e = p / (g - 1) + 0.5 * (u ** 2 + v ** 2) * rho

    # Create state vectors
    Q = np.array([rho, rho * u, rho * v, e])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.nx):
                block.state.U[i, j, :] = Q

        block.state.non_dim()

def explosion_trapezoid(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rhoL = 4.6968
    pL = 404400.0
    uL = 0.0
    vL = 0.0
    eL = pL / (g - 1) + rhoL * (uL ** 2 + vL ** 2) / 2

    # Low pressure zone
    rhoR = 1.1742
    pR = 101100.0
    uR = 0.00
    vR = 0.0
    eR = pR / (g - 1) + rhoR * (uR ** 2 + vR ** 2) / 2

    # Create state vectors
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.ny):
                if (-0.75 <= block.mesh.x[i, j] <= -0.25 and -0.75 <= block.mesh.y[i, j] <= -0.25) or \
                   ( 0.25 <= block.mesh.x[i, j] <=  0.75 and  0.25 <= block.mesh.y[i, j] <=  0.75) or \
                   (-0.75 <= block.mesh.x[i, j] <= -0.25 and  0.25 <= block.mesh.y[i, j] <=  0.75) or \
                   ( 0.25 <= block.mesh.x[i, j] <=  0.75 and -0.75 <= block.mesh.y[i, j] <= -0.25):
                    block.state.U[i, j, :] = QL
                else:
                    block.state.U[i, j, :] = QR

        block.state.non_dim()

def explosion_3(blocks, **kwargs):

    if 'g' not in kwargs.keys():
        raise KeyError('Parameter g (gamma) must be passed to the explosion IC function.')

    # Gamma
    g = kwargs['g']

    # High pressure zone
    rhoL = 4.6968
    pL = 404400.0
    uL = 0.0
    vL = 0.0
    eL = pL / (g - 1) + rhoL * (uL ** 2 + vL ** 2) / 2

    # Low pressure zone
    rhoR = 1.1742
    pR = 101100.0
    uR = 0.00
    vR = 0.0
    eR = pR / (g - 1) + rhoR * (uR ** 2 + vR ** 2) / 2

    # Create state vectors
    QL = np.array([rhoL, rhoL * uL, rhoL * vL, eL])
    QR = np.array([rhoR, rhoR * uR, rhoR * vR, eR])

    # Fill state vector in each block
    for block in blocks:
        for i in range(block.mesh.ny):
            for j in range(block.mesh.ny):
                if (-0.25 <= block.mesh.x[i, j] <= 0.25 and -0.75 <= block.mesh.y[i, j] <= 0.75) or \
                   (-0.75 <= block.mesh.x[i, j] <= 0.75 and -0.25 <= block.mesh.y[i, j] <= 0.25):
                    block.state.U[i, j, :] = QL
                else:
                    block.state.U[i, j, :] = QR

        block.state.non_dim()
