import numba
from numba import int32, float32
from numba.core.types import ListType, DictType
from numba.typed import Dict as nDict
from numba.typed import List as nList
from numba.experimental import jitclass
from numba.core.types import string as nstr
from pyHype.mesh.mesh_builder import BlockDescription


_block_description_type = BlockDescription.class_type.instance_type
_mesh_input_type = numba.types.DictType(int32, _block_description_type)

_PROBLEM_INPUT_SPEC = [('problem_type', nstr),
                       ('IC_type', nstr),
                       ('realplot', int32),
                       ('time_it', int32),
                       ('t_final', float32),
                       ('time_integrator', nstr),
                       ('CFL', float32),
                       ('flux_function', nstr),
                       ('finite_volume_method', nstr),
                       ('flux_limiter', nstr),
                       ('gamma', float32),
                       ('R', float32),
                       ('rho_inf', int32),
                       ('a_inf', int32),
                       ('n', int32),
                       ('nx', int32),
                       ('ny', int32),
                       ('mesh_name', nstr),
                       ('mesh_inputs', _mesh_input_type)]


@jitclass(_PROBLEM_INPUT_SPEC)
class ProblemInput:
    def __init__(self, _list, _int, _flt, _str, _mesh):
        """
        Sets required input parametes from input parameter dict. Initialized values to default, with the correct type
        """

        # General parameters
        self.problem_type = _str['problem_type']
        self.IC_type = _str['IC_type']
        self.realplot = _int['realplot']
        self.time_it = _int['time_it']
        self.t_final = _flt['t_final']

        # Numerical method parameters
        self.time_integrator = _str['time_integrator']
        self.CFL = _flt['CFL']
        self.flux_function = _str['flux_function']
        self.finite_volume_method = _str['finite_volume_method']
        self.flux_limiter = _str['flux_limiter']

        # Thermodynamic parameters
        self.gamma = _flt['gamma']
        self.R = _flt['R']
        self.rho_inf = _flt['rho_inf']
        self.a_inf = _flt['a_inf']

        # Mesh parameters
        self.n = _int['nx'] * _int['ny']
        self.nx = _int['nx']
        self.ny = _int['ny']
        self.mesh_name = _str['mesh_name']
        self.mesh_inputs = _mesh


def build(inputs_: dict, mesh_inputs: DictType):
    _lst, _int, _flt, _str = dict_to_numbadict(inputs_)
    return ProblemInput(_lst, _int, _flt, _str, mesh_inputs)


def dict_to_numbadict(dict_: dict):

    _lst = ListType(int32)
    _list_items = nDict.empty(key_type=nstr, value_type=_lst)
    _int_items = nDict.empty(key_type=nstr, value_type=int32)
    _flt_items = nDict.empty(key_type=nstr, value_type=float32)
    _str_items = nDict.empty(key_type=nstr, value_type=nstr)

    for key, value in dict_.items():
        if isinstance(value, list):
            _list_items[key] = nList(value)
        elif isinstance(value, int):
            _int_items[key] = value
        elif isinstance(value, float):
            _flt_items[key] = value
        elif isinstance(value, str):
            _str_items[key] = value

    return _list_items, _int_items, _flt_items, _str_items
