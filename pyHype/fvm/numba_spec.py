from numba.core.types import string as nstr, int32, float64
from numba.core.types import DictType
from pyHype.mesh.mesh_builder import BlockDescription
from pyHype.input.input_file_builder import propblem_input_type

block_description_type = BlockDescription.class_type.instance_type
mesh_inputs_type = DictType(int32, block_description_type)

SecondOrderLimited_SPEC = [('inputs', propblem_input_type),
                           ('global_nBLK', int32),
                           ('nx', int32),
                           ('ny', int32),
                           ('Flux_X', float64[:, :]),
                           ('Flux_Y', float64[:, :]),
                           ]
