from numba.core.types import string as nstr, int32, float64
from numba.core.types import DictType
from pyHype.mesh.mesh_builder import BlockDescription


block_description_type = BlockDescription.class_type.instance_type
mesh_inputs_type = DictType(int32, block_description_type)

PROBLEM_INPUT_SPEC = [('problem_type', nstr),
                       ('IC_type', nstr),
                       ('realplot', int32),
                       ('time_it', int32),
                       ('t_final', float64),
                       ('time_integrator', nstr),
                       ('CFL', float64),
                       ('flux_function', nstr),
                       ('finite_volume_method', nstr),
                       ('flux_limiter', nstr),
                       ('gamma', float64),
                       ('R', float64),
                       ('rho_inf', int32),
                       ('a_inf', int32),
                       ('n', int32),
                       ('nx', int32),
                       ('ny', int32),
                       ('mesh_name', nstr),
                       ('mesh_inputs', mesh_inputs_type)]
