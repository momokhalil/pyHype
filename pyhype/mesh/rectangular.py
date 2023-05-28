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
from pyhype.mesh.base import QuadMeshGenerator

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
import numpy as np


class RectagularMeshGenerator:
    @staticmethod
    def generate(
        BCE: [str],
        BCW: [str],
        BCN: [str],
        BCS: [str],
        east: float,
        west: float,
        north: float,
        south: float,
        n_blocks_horizontal: int,
        n_blocks_vertical: int,
    ):
        if east <= west:
            raise ValueError(f"East value {east} must be larger than west {west}")

        if north <= south:
            raise ValueError(f"North value {north} must be larger than south {south}")

        top_bot_y_coords = np.ones(n_blocks_horizontal + 1)
        top_bot_x_coords = np.linspace(west, east, n_blocks_horizontal + 1)
        east_west_x_coords = np.ones(n_blocks_vertical + 1)
        east_west_y_coords = np.linspace(south, north, n_blocks_vertical + 1)
        return QuadMeshGenerator(
            nx_blk=n_blocks_horizontal,
            ny_blk=n_blocks_vertical,
            BCE=BCE,
            BCW=BCW,
            BCN=BCN,
            BCS=BCS,
            top_x=top_bot_x_coords,
            bot_x=top_bot_x_coords,
            top_y=north * top_bot_y_coords,
            bot_y=south * top_bot_y_coords,
            left_x=west * east_west_x_coords,
            right_x=east * east_west_x_coords,
            left_y=east_west_y_coords,
            right_y=east_west_y_coords,
        )
