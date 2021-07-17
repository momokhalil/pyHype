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
              'NeighborW': 0,
              'NeighborN': 2,
              'NeighborS': 0,
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
              'NeighborW': 0,
              'NeighborN': 0,
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
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
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
              'NeighborE': 0,
              'NeighborW': 1,
              'NeighborN': 3,
              'NeighborS': 0,
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
              'NeighborE': 0,
              'NeighborW': 0,
              'NeighborN': 2,
              'NeighborS': 0,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'Outflow',
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
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'Outflow',
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
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
              'NeighborS': 0,
              'BCTypeE': 'Outflow',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1,
            2: block2,
            3: block3}

def ramp_three_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NE': [5.5, 5],
              'NW': [0, 5],
              'SE': [3, 0],
              'SW': [0, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 0,
              'NeighborW': 0,
              'NeighborN': 2,
              'NeighborS': 0,
              'BCTypeE': 'Slipwall',
              'BCTypeW': 'Outflow',
              'BCTypeN': 'None',
              'BCTypeS': 'Slipwall'}

    block2 = {'nBLK': 2,
              'NE': [5.5, 10],
              'NW': [0, 10],
              'SE': [5.5, 5],
              'SW': [0, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 3,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 1,
              'BCTypeE': 'None',
              'BCTypeW': 'Outflow',
              'BCTypeN': 'Slipwall',
              'BCTypeS': 'None'}

    block3 = {'nBLK': 3,
              'NE': [10, 10],
              'NW': [5.5, 10],
              'SE': [10, 5],
              'SW': [5.5, 5],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
              'NeighborS': 0,
              'BCTypeE': 'Outflow',
              'BCTypeW': 'None',
              'BCTypeN': 'Slipwall',
              'BCTypeS': 'Slipwall'}

    return {1: block1,
            2: block2,
            3: block3}

def square_ten_by_ten_one_block(nx, ny, nghost):
    block1 = {'nBLK': 1,
              'NW': [0, 5], 'NE': [5, 5],
              'SW': [0, 0], 'SE': [5, 0],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 0,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 0,
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
              'NeighborE': 0,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 0,
              'BCTypeE': 'Reflection',
              'BCTypeW': 'Reflection',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    return {1: block1}

def chamber_skewed(nx, ny, nghost):

    block1 = {'nBLK': 1,
              'NW': [-2, 1], 'NE': [-1, 1],
              'SW': [-3, -1], 'SE': [-1, -1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 2,
              'NeighborW': 0,
              'NeighborN': 0,
              'NeighborS': 0,
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
              'NeighborN': 0,
              'NeighborS': 0,
              'BCTypeE': 'None',
              'BCTypeW': 'None',
              'BCTypeN': 'Reflection',
              'BCTypeS': 'Reflection'}

    block3 = {'nBLK': 3,
              'NW': [1, 1], 'NE': [2, 1],
              'SW': [1, -1], 'SE': [3, -1],
              'nx': nx,
              'ny': ny,
              'n': nx * ny,
              'nghost': nghost,
              'NeighborE': 0,
              'NeighborW': 2,
              'NeighborN': 0,
              'NeighborS': 0,
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
                  'ramp_three_block': ramp_three_block,
                  'chamber': chamber,
                  'chamber_skewed': chamber_skewed}


