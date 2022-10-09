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

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"


def step_ten_block():

    block1 = {
        "nBLK": 1,
        "NW": [0, 1],
        "NE": [3, 1],
        "SW": [0, 0],
        "SE": [3, 0],
        "NeighborE": None,
        "NeighborW": None,
        "NeighborN": 2,
        "NeighborS": None,
        "BCTypeE": "Slipwall",
        "BCTypeW": "InletDirichlet",
        "BCTypeN": "None",
        "BCTypeS": "Slipwall",
    }

    block2 = {
        "nBLK": 2,
        "NW": [0, 2],
        "NE": [3, 2],
        "SW": [0, 1],
        "SE": [3, 1],
        "NeighborE": 7,
        "NeighborW": None,
        "NeighborN": 3,
        "NeighborS": 1,
        "BCTypeE": "None",
        "BCTypeW": "InletDirichlet",
        "BCTypeN": "None",
        "BCTypeS": "None",
    }

    block3 = {
        "nBLK": 3,
        "NW": [0, 3],
        "NE": [3, 3],
        "SW": [0, 2],
        "SE": [3, 2],
        "NeighborE": 6,
        "NeighborW": None,
        "NeighborN": 4,
        "NeighborS": 2,
        "BCTypeE": "None",
        "BCTypeW": "InletDirichlet",
        "BCTypeN": "None",
        "BCTypeS": "None",
    }

    block4 = {
        "nBLK": 4,
        "NW": [0, 4],
        "NE": [3, 4],
        "SW": [0, 3],
        "SE": [3, 3],
        "NeighborE": 5,
        "NeighborW": None,
        "NeighborN": None,
        "NeighborS": 3,
        "BCTypeE": "None",
        "BCTypeW": "InletDirichlet",
        "BCTypeN": "Slipwall",
        "BCTypeS": "None",
    }

    block5 = {
        "nBLK": 5,
        "NW": [3, 4],
        "NE": [6, 4],
        "SW": [3, 3],
        "SE": [6, 3],
        "NeighborE": 10,
        "NeighborW": 4,
        "NeighborN": None,
        "NeighborS": 6,
        "BCTypeE": "None",
        "BCTypeW": "None",
        "BCTypeN": "Slipwall",
        "BCTypeS": "None",
    }

    block6 = {
        "nBLK": 6,
        "NW": [3, 3],
        "NE": [6, 3],
        "SW": [3, 2],
        "SE": [6, 2],
        "NeighborE": 9,
        "NeighborW": 3,
        "NeighborN": 5,
        "NeighborS": 7,
        "BCTypeE": "None",
        "BCTypeW": "None",
        "BCTypeN": "None",
        "BCTypeS": "None",
    }

    block7 = {
        "nBLK": 7,
        "NW": [3, 2],
        "NE": [6, 2],
        "SW": [3, 1],
        "SE": [6, 1],
        "NeighborE": 8,
        "NeighborW": 2,
        "NeighborN": 6,
        "NeighborS": None,
        "BCTypeE": "None",
        "BCTypeW": "None",
        "BCTypeN": "None",
        "BCTypeS": "Slipwall",
    }

    block8 = {
        "nBLK": 8,
        "NW": [6, 2],
        "NE": [9, 2],
        "SW": [6, 1],
        "SE": [9, 1],
        "NeighborE": None,
        "NeighborW": 7,
        "NeighborN": 9,
        "NeighborS": None,
        "BCTypeE": "OutletDirichlet",
        "BCTypeW": "None",
        "BCTypeN": "None",
        "BCTypeS": "Slipwall",
    }

    block9 = {
        "nBLK": 9,
        "NW": [6, 3],
        "NE": [9, 3],
        "SW": [6, 2],
        "SE": [9, 2],
        "NeighborE": None,
        "NeighborW": 6,
        "NeighborN": 10,
        "NeighborS": 8,
        "BCTypeE": "OutletDirichlet",
        "BCTypeW": "None",
        "BCTypeN": "None",
        "BCTypeS": "None",
    }

    block10 = {
        "nBLK": 10,
        "NW": [6, 4],
        "NE": [9, 4],
        "SW": [6, 3],
        "SE": [9, 3],
        "NeighborE": None,
        "NeighborW": 5,
        "NeighborN": None,
        "NeighborS": 9,
        "BCTypeE": "OutletDirichlet",
        "BCTypeW": "None",
        "BCTypeN": "Slipwall",
        "BCTypeS": "None",
    }

    return {
        1: block1,
        2: block2,
        3: block3,
        4: block4,
        5: block5,
        6: block6,
        7: block7,
        8: block8,
        9: block9,
        10: block10,
    }
