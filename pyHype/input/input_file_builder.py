from numba import int32, float64
from numba.typed import Dict as nDict
from numba.typed import List as nList
import pyHype.input.numba_spec as ns
from numba.experimental import jitclass
from numba.core.types import string as nstr
from numba.core.types import ListType, DictType


@jitclass(ns.PROBLEM_INPUT_SPEC)
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
        self.meshinputss = _mesh


propblem_input_type = ProblemInput.class_type.instance_type


def build(inputs_: dict, meshinputss: DictType):
    _lst, _int, _flt, _str = dict_to_numbadict(inputs_)
    return ProblemInput(_lst, _int, _flt, _str, meshinputss)


def dict_to_numbadict(dict_: dict):

    _lst = ListType(int32)
    _list_items = nDict.empty(key_type=nstr, value_type=_lst)
    _int_items = nDict.empty(key_type=nstr, value_type=int32)
    _flt_items = nDict.empty(key_type=nstr, value_type=float64)
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
