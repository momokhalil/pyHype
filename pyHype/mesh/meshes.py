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
import os
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np

# Meshes
def square_ten_by_ten_four_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NE': [5, 5],
              'NW': [0, 5],
              'SE': [5, 0],
              'SW': [0, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NE': [5, 10],
              'NW': [0, 10],
              'SE': [5, 5],
              'SW': [0, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NE': [10, 10],
              'NW': [5, 10],
              'SE': [10, 5],
              'SW': [5, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': 4,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'None'}

    block4 = {'nBLK': 4,
              'NE': [10, 5],
              'NW': [5, 5],
              'SE': [10, 0],
              'SW': [5, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4}

# Meshes
def step_three_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NE': [5, 5],
              'NW': [0, 5],
              'SE': [5, 0],
              'SW': [0, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NE': [5, 10],
              'NW': [0, 10],
              'SE': [5, 5],
              'SW': [0, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NE': [10, 10],
              'NW': [5, 10],
              'SE': [10, 5],
              'SW': [5, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3}

def ramp_six_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 2], 'NE': [2, 2],
              'SW': [0, 0], 'SE': [1, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
              'BCTypeE': 'Slipwall',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'None',
              'BCTypeS': 'Slipwall'}

    block2 = {'nBLK': 2,
              'NW': [0, 4], 'NE': [2, 4],
              'SW': [0, 2], 'SE': [2, 2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 5,
              'NeighborW': None,
              'NeighborN': 3,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'None',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NW': [0, 6], 'NE': [2, 6],
              'SW': [0, 4], 'SE': [2, 4],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': 2,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Slipwall',
              'BCTypeS': 'None'}

    block4 = {'nBLK': 4,
              'NW': [2, 6], 'NE': [4, 6],
              'SW': [2, 4], 'SE': [4, 4],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 3,
              'NeighborN': None,
              'NeighborS': 5,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'Slipwall',
              'BCTypeS': 'None'}

    block5 = {'nBLK': 5,
              'NW': [2, 4], 'NE': [4, 4],
              'SW': [2, 2], 'SE': [4, 2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': 4,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'None',
              'BCTypeS': 'Slipwall'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4,
            5: block5}

def ramp_two_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 2], 'NE': [2, 2],
              'SW': [0, 0], 'SE': [2 / np.tan(61 * np.pi / 180), 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'None',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [0, 4], 'NE': [2, 4],
              'SW': [0, 2], 'SE': [2, 2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': 1,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'None'}

    return {1: block1,
            2: block2}

def shallow_ramp_four_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 2], 'NE': [2, 2],
              'SW': [0, 0], 'SE': [2, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': None,
              'NeighborN': 2,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'None',
              'BCTypeS': 'Slipwall'}

    block2 = {'nBLK': 2,
              'NW': [0, 4], 'NE': [2, 4],
              'SW': [0, 2], 'SE': [2, 2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NW': [2, 4], 'NE': [4, 4],
              'SW': [2, 2], 'SE': [4, 2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': 4,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'None'}

    block4 = {'nBLK': 4,
              'NW': [2, 2], 'NE': [4, 2],
              'SW': [2, 0], 'SE': [4, 2 * np.tan(15 * np.pi / 180)],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'None',
              'BCTypeS': 'Slipwall'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4
            }

def shallow_ramp_two_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 1], 'NE': [1, 1],
              'SW': [0, 0], 'SE': [1, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [1, 1], 'NE': [2, 1],
              'SW': [1, 0], 'SE': [2, 1 * np.tan(15 * np.pi / 180)],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2
            }

def shallow_ramp_multiangle_three_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 1], 'NE': [1, 1],
              'SW': [0, 0], 'SE': [1, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [1, 1], 'NE': [2, 1],
              'SW': [1, 0], 'SE': [2, 1 * np.tan(7 * np.pi / 180)],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [2, 1], 'NE': [3, 1],
              'SW': [2, 1 * np.tan(7 * np.pi / 180)], 'SE': [3, 1 * np.tan(20 * np.pi / 180)],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'OutletDirichlet',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3
            }


def ramp_channel(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [-0.2, 1], 'NE': [0.2, 1],
              'SW': [-0.2, 0], 'SE': [0.4, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [0.2, 1], 'NE': [0.8, 0.80],
              'SW': [0.4, 0], 'SE': [0.9, 0.1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [0.8, 0.80], 'NE': [1.3, 0.75],
              'SW': [0.9, 0.1], 'SE': [1.3, 0.25],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block4 = {'nBLK': 4,
              'NW': [1.3, 0.75], 'NE': [1.55, 0.75],
              'SW': [1.3, 0.25], 'SE': [1.60, 0.5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 5,
              'NeighborW': 3,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block5 = {'nBLK': 5,
              'NW': [1.55, 0.75], 'NE': [1.9, 0.75],
              'SW': [1.60, 0.50], 'SE': [1.9, 0.58],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 6,
              'NeighborW': 4,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block6 = {'nBLK': 6,
              'NW': [1.9, 0.75], 'NE': [2.2, 0.75],
              'SW': [1.9, 0.58], 'SE': [2.2, 0.58],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 5,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    """block6 = {'nBLK': 6,
              'NW': [0, 0], 'NE': [0.5, 0],
              'SW': [0, -1], 'SE': [0.5, -1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 7,
              'NeighborW': None,
              'NeighborN': 1,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}"""

    block7 = {'nBLK': 7,
              'NW': [0.5, 0], 'NE': [1.1, -0.1],
              'SW': [0.5, -1], 'SE': [1.0, -0.85],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 8,
              'NeighborW': 6,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block8 = {'nBLK': 8,
              'NW': [1.1, -0.1], 'NE': [1.5, -0.25],
              'SW': [1.0, -0.85], 'SE': [1.5, -0.8],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 9,
              'NeighborW': 7,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block9 = {'nBLK': 9,
              'NW': [1.5, -0.25], 'NE': [1.85, -0.25],
              'SW': [1.5, -0.8], 'SE': [1.85, -0.8],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 10,
              'NeighborW': 8,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block10 = {'nBLK': 10,
               'NW': [1.85, -0.25], 'NE': [2.2, -0.25],
               'SW': [1.85, -0.8], 'SE': [2.2, -0.8],
               'nx': nx,
               'ny': ny,
               'n': nx * ny,
               'nghost': nghost,
               'NeighborE': None,
               'NeighborW': 9,
               'NeighborN': None,
               'NeighborS': None,
               'BCTypeE': 'OutletDirichlet',
               'BCTypeW': 'None',
               'BCTypeN': 'Reflection',
               'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4,
            5: block5,
            6: block6,
            }

def long_nozzle(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0.0, 0.6], 'NE': [0.25, 0.6],
              'SW': [0.0, -0.6], 'SE': [0.25, -0.6],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [0.25, 0.6], 'NE': [0.5, 0.40],
              'SW': [0.25, -0.6], 'SE': [0.5, -0.40],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [0.5, 0.40], 'NE': [0.75, 0.2],
              'SW': [0.5, -0.40], 'SE': [0.75, -0.2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block4 = {'nBLK': 4,
              'NW': [0.75, 0.2], 'NE': [0.875, 0.17],
              'SW': [0.75, -0.2], 'SE': [0.875, -0.17],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 5,
              'NeighborW': 3,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block5 = {'nBLK': 5,
              'NW': [0.875, 0.17], 'NE': [1.00, 0.2],
              'SW': [0.875, -0.17], 'SE': [1.00, -0.2],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 6,
              'NeighborW': 4,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block6 = {'nBLK': 6,
              'NW': [1.00, 0.2], 'NE': [1.125, 0.26],
              'SW': [1.00, -0.2], 'SE': [1.125, -0.26],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 7,
              'NeighborW': 5,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block7 = {'nBLK': 7,
              'NW': [1.125, 0.26], 'NE': [1.25, 0.34],
              'SW': [1.125, -0.26], 'SE': [1.25, -0.34],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 8,
              'NeighborW': 6,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block8 = {'nBLK': 8,
              'NW': [1.25, 0.34], 'NE': [1.5, 0.49],
              'SW': [1.25, -0.34], 'SE': [1.5, -0.49],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 9,
              'NeighborW': 7,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block9 = {'nBLK': 9,
              'NW': [1.5, 0.49], 'NE': [1.75, 0.6],
              'SW': [1.5, -0.49], 'SE': [1.75, -0.6],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 10,
              'NeighborW': 8,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block10 = {'nBLK': 10,
               'NW': [1.75, 0.6], 'NE': [2.0, 0.69],
               'SW': [1.75, -0.6], 'SE': [2.0, -0.69],
               'nx': nx,
               'ny': ny,
               'n': nx * ny,
               'nghost': nghost,
               'NeighborE': None,
               'NeighborW': 9,
               'NeighborN': None,
               'NeighborS': None,
               'BCTypeE': 'OutletDirichlet',
               'BCTypeW': 'None',
               'BCTypeN': 'Reflection',
               'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4,
            5: block5,
            6: block6,
            7: block7,
            8: block8,
            9: block9,
            10: block10
            }

def ramjet(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [-0.5, 1], 'NE': [0.0, 1],
              'SW': [-0.5, 0], 'SE': [0.0, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'InletDirichlet',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [0.0, 1], 'NE': [0.5, 0.94],
              'SW': [0.0, 0], 'SE': [0.5, 0.00],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [0.5, 0.94], 'NE': [1.0, 0.85],
              'SW': [0.5, 0.00], 'SE': [1.0, 0.00],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block4 = {'nBLK': 4,
              'NW': [1.0, 0.85], 'NE': [1.50, 0.7],
              'SW': [1.0, 0.00], 'SE': [1.50, 0.0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 5,
              'NeighborW': 3,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block5 = {'nBLK': 5,
              'NW': [1.50, 0.7], 'NE': [2.0, 0.63],
              'SW': [1.50, 0.0], 'SE': [2.0, 0.1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 6,
              'NeighborW': 4,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block6 = {'nBLK': 6,
              'NW': [2.0, 0.63], 'NE': [2.5, 0.60],
              'SW': [2.0, 0.1], 'SE': [2.5, 0.25],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 7,
              'NeighborW': 5,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block7 = {'nBLK': 7,
              'NW': [2.5, 0.60], 'NE': [3.0, 0.60],
              'SW': [2.5, 0.25], 'SE': [3.0, 0.32],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 8,
              'NeighborW': 6,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block8 = {'nBLK': 8,
              'NW': [3.0, 0.60], 'NE': [3.5, 0.60],
              'SW': [3.0, 0.32], 'SE': [3.5, 0.34],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 9,
              'NeighborW': 7,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block9 = {'nBLK': 9,
              'NW': [3.5, 0.60], 'NE': [4.0, 0.60],
              'SW': [3.5, 0.34], 'SE': [4.0, 0.34],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 8,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'OutletDirichlet',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4,
            5: block5,
            6: block6,
            7: block7,
            8: block8,
            9: block9,
            }

def square_ten_by_ten_one_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 5], 'NE': [5, 5],
              'SW': [0, 0], 'SE': [5, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1}


def chamber(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 20], 'NE': [10, 20],
              'SW': [0, 0],   'SE': [10, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1}

def chamber_skewed_2(nx, ny, nghost):

    block1 = {'nBLK': 1,
              'NW': [-4.5, 0.0], 'NE': [-3.0, 0.0],
              'SW': [-5.0, -1.0], 'SE': [-3.0, -1.5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [-3.0, 0.0], 'NE': [-1.0, 1.0],
              'SW': [-3.0, -1.5], 'SE': [-1.0, -1.0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [-1.0, 1.0], 'NE': [1.0, 1.0],
              'SW': [-1.0, -1.0], 'SE': [1.0, -1.0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 4,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block4 = {'nBLK': 4,
              'NW': [1.0, 1.0], 'NE': [3, 0.0],
              'SW': [1.0, -1.0], 'SE': [3, -1.5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 5,
              'NeighborW': 3,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block5 = {'nBLK': 5,
              'NW': [3.0, 0.0], 'NE': [4.5, 0.0],
              'SW': [3.0, -1.5], 'SE': [5.0, -1.0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 4,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3,
            4: block4,
            5: block5
            }

def chamber_skewed(nx, ny, nghost):

    block1 = {'nBLK': 1,
              'NW': [-2.5, 1.5], 'NE': [-1, 1],
              'SW': [-2.5, 0.0], 'SE': [-1, -1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': None,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block2 = {'nBLK': 2,
              'NW': [-1, 1], 'NE': [1, 1],
              'SW': [-1, -1], 'SE': [1, -1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 1,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [1, 1], 'NE': [2.5, 0.0],
              'SW': [1, -1], 'SE': [2.5, -1.5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': None,
              'NeighborW': 2,
              'NeighborN': None,
              'NeighborS': None,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3}


DEFINED_MESHES = {'square_ten_by_ten_four_block': square_ten_by_ten_four_block,
                  'square_ten_by_ten_one_block': square_ten_by_ten_one_block,
                  'step_three_block': step_three_block,
                  'ramp_two_block': ramp_two_block,
                  'ramp_six_block': ramp_six_block,
                  'shallow_ramp_four_block': shallow_ramp_four_block,
                  'shallow_ramp_two_block': shallow_ramp_two_block,
                  'shallow_ramp_multiangle_three_block': shallow_ramp_multiangle_three_block,
                  'ramp_channel': ramp_channel,
                  'ramjet': ramjet,
                  'long_nozzle': long_nozzle,
                  'chamber': chamber,
                  'chamber_skewed': chamber_skewed,
                  'chamber_skewed_2': chamber_skewed_2}


